import sys
import os
import json
from pathlib import Path
import gym
from gym.core import Wrapper

current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from prismatic_openvla.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic_openvla.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic_openvla.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from absl import app, flags
import glob
import re

import peft

from manipulator_gym.manipulator_env import ManipulatorEnv
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.utils.gym_wrappers import ClipActionBoxBoundary, ConvertState2Proprio, ResizeObsImageWrapper
from vla_adapter_action_head_utils import L1RegressionActionHead

FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "localhost", "IP of robot server")
flags.DEFINE_integer("port", 5556, "Port of robot server")
flags.DEFINE_string("checkpoint_dir", None, "Path to checkpoint")
flags.DEFINE_bool("show_img", False, "Visualize images")
flags.DEFINE_string("text_cond", "put the banana on the plate", "Task description")
flags.DEFINE_bool("clip_actions", False, "Clip xyz/rpy movements")

DEVICE = "cuda:0"
ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_OPEN_LOOP_STEPS = 8


@dataclass
class GenerateConfig:
    checkpoint_dir: str
    use_proprio: bool = True
    use_wrist_cam: bool = True
    use_l1_regression: bool = True
    num_open_loop_steps: int = NUM_OPEN_LOOP_STEPS
    unnorm_key: str = "my_bridge"


class ProprioProjector(nn.Module):
    def __init__(self, llm_dim: int, proprio_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act = nn.GELU()

    def forward(self, proprio):
        x = self.fc1(proprio)
        x = self.act(x)
        x = self.fc2(x)
        return x


def normalize_proprio(proprio, norm_stats):
    mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
    high = np.array(norm_stats["q99"])
    low = np.array(norm_stats["q01"])

    return np.clip(
        np.where(mask, 2*(proprio-low)/(high-low+1e-8)-1, proprio),
        -1.0,
        1.0
    )


def get_action(cfg, vla, processor, obs, task_label,
               action_head=None, proprio_projector=None):

    with torch.inference_mode():

        images = [obs["full_image"]]

        if cfg.use_wrist_cam and "wrist_image" in obs:
            images.append(obs["wrist_image"])

        primary_image = images.pop(0)

        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        if images:
            wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in images]
            px = [inputs["pixel_values"]] + [wi["pixel_values"] for wi in wrist_inputs]
            inputs["pixel_values"] = torch.cat(px, dim=1)

        proprio = None

        if cfg.use_proprio and "state" in obs:
            proprio = normalize_proprio(obs["state"], vla.norm_stats[cfg.unnorm_key]["proprio"])
            proprio = torch.tensor(proprio, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)

        if action_head is None:
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head
            )

        return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]


def main(_):

    cfg = GenerateConfig(checkpoint_dir=FLAGS.checkpoint_dir)

    interface = ActionClientInterface(host=FLAGS.ip, port=FLAGS.port)

    env = ManipulatorEnv(
        manipulator_interface=interface,
        use_wrist_cam=cfg.use_wrist_cam
    )

    env = ClipActionBoxBoundary(
        env, workspace_boundary=[[-float('inf'), -float('inf'), -float('inf')], [float('inf'), float('inf'), float('inf')]]
    )
    
    render_size = (256, 256)
    env = ConvertState2Proprio(env)
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": render_size, "image_wrist": render_size}
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    config = OpenVLAConfig.from_pretrained(cfg.checkpoint_dir)

    processor = AutoProcessor.from_pretrained(
        cfg.checkpoint_dir,
        trust_remote_code=True
    )

    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE)

    vla.vision_backbone.set_num_images_in_input(2)

    vla.eval()

    lora_path = os.path.join(cfg.checkpoint_dir, "lora_adapter")

    if os.path.isdir(lora_path):
        vla = peft.PeftModel.from_pretrained(vla, lora_path)
        vla = vla.merge_and_unload()

    stats_path = os.path.join(cfg.checkpoint_dir, "dataset_statistics.json")

    with open(stats_path) as f:
        vla.norm_stats = json.load(f)

    proprio_projector = None

    if cfg.use_proprio:

        proprio_projector = ProprioProjector(
            llm_dim=vla.llm_dim,
            proprio_dim=PROPRIO_DIM
        )

        proprio_projector = proprio_projector.to(DEVICE).to(torch.bfloat16)
        proprio_projector.eval()

    action_head = None

    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=ACTION_DIM,
            use_pro_version=True
        )


        pattern = os.path.join(cfg.checkpoint_dir, "action_head--*_checkpoint.pt")
        ah_files = glob.glob(pattern)

        if not ah_files:

            default_path = os.path.join(cfg.checkpoint_dir, "action_head.pt")
            if os.path.exists(default_path):
                ah_path = default_path
            else:
                raise FileNotFoundError(f"在 {cfg.checkpoint_dir} 中没找到任何 action_head 文件！")
        else:

            ah_files.sort(key=lambda x: int(re.search(r'--(\d+)_', x).group(1)))
            ah_path = ah_files[-1]
            print(f"--- Action Head: {os.path.basename(ah_path)} ---")


        state_dict = torch.load(ah_path, map_location=DEVICE)
        new_state_dict = {}

        for k, v in state_dict.items():

            name = k[7:] if k.startswith("module.") else k

            if not name.startswith("model."):
                name = "model." + name
            new_state_dict[name] = v

        action_head.load_state_dict(new_state_dict)
        action_head = action_head.to(DEVICE).to(torch.bfloat16).eval()

    try:

        for _ in range(100):

            obs, _ = env.reset()

            action_queue = deque(maxlen=cfg.num_open_loop_steps)

            for _ in range(100):

                if "state" not in obs:
                    raw_pose = interface.eef_pose[:6]
                    gripper = interface.gripper_state
                    obs["state"] = np.concatenate([np.zeros(6), [0], np.zeros(1)], dtype=np.float32)

                img_primary = cv2.cvtColor(obs["image_primary"], cv2.COLOR_RGB2BGR)

                if cfg.use_wrist_cam:
                    img_wrist = cv2.cvtColor(obs["image_wrist"], cv2.COLOR_RGB2BGR)

                if FLAGS.show_img:
                    cv2.imshow("primary", img_primary)

                    if cfg.use_wrist_cam:
                        cv2.imshow("wrist", img_wrist)

                    cv2.waitKey(10)

                obs_policy = {
                    "full_image": Image.fromarray(img_primary),
                    "state": obs["state"]
                }

                if cfg.use_wrist_cam:
                    obs_policy["wrist_image"] = Image.fromarray(img_wrist)

                if len(action_queue) == 0:
                    actions = get_action(
                        cfg,
                        vla,
                        processor,
                        obs_policy,
                        FLAGS.text_cond,
                        action_head,
                        proprio_projector
                    )
                    action_queue.extend(actions)

                action = action_queue.popleft()

                if FLAGS.clip_actions:
                    action[:6] = np.clip(action[:6], -0.02, 0.02)

                obs, _, done, trunc, _ = env.step(action)

                if done:
                    break

    except KeyboardInterrupt:
        env.reset()


if __name__ == "__main__":
    app.run(main)