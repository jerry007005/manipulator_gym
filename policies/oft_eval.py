"""
This script to eval OpenVLA model on bridge data robot setup.
"""


import time
import numpy as np
from absl import app, flags, logging
import cv2
from collections import deque
from PIL import Image
import os

from manipulator_gym.manipulator_env import ManipulatorEnv
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.utils.gym_wrappers import ClipActionBoxBoundary, ConvertState2Proprio, ResizeObsImageWrapper


from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional

from openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from openvla_oft.prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "localhost", "IP address of the robot server.")
flags.DEFINE_integer("port", 5556, "Port of the manipulator server.")
flags.DEFINE_bool("show_img", False, "Whether to visualize the images or not.")
flags.DEFINE_string(
    "text_cond", "put the banana on the plate", "Language prompt for the task."
)
flags.DEFINE_string("lora_adapter_dir", None, "Path to the LORA adapter directory.")
flags.DEFINE_bool("clip_actions", False, "Clip actions to 0.02")
flags.DEFINE_string(
    "dataset_stats",
    "my_bridge",
    "Path to the dataset stats json file, default to brige_orig.",
)
# Example lora_adapter_dir: "adapter-tmp/openvla-7b+serl_demos+b4+lr-2e-05+lora-r32+dropout-0.0+q-4bit/"


# np decimal printout to 2 decimal places
np.set_printoptions(precision=2, suppress=True)
device = "cuda:0"

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)

    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization


    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


class FrameSaver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.idx = 0

    def save(self, frame):
        filename = os.path.join(self.save_dir, f"{self.idx:06d}.png")
        cv2.imwrite(filename, frame)
        self.idx += 1




def main(_):
    interface = ActionClientInterface(host=FLAGS.ip, port=FLAGS.port)
    env = ManipulatorEnv(
        manipulator_interface=interface,
        use_wrist_cam = True,
        # manipulator_interface=ManipulatorInterface(), # for testing
    )  # default doesn't use wrist cam
    # NOTE: using the kitchen sink setup boundary of https://github.com/simpler-env/SimplerEnv
    env = ClipActionBoxBoundary(
        env, workspace_boundary=[[-float('inf'), -float('inf'), -float('inf')], [float('inf'), float('inf'), float('inf')]]
    )
    
    render_size = (256, 256)
    env = ConvertState2Proprio(env)
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": render_size, "image_wrist": render_size}
    )

    cfg = GenerateConfig(
        pretrained_checkpoint = "/home/yzho684/haochuan/manipulator_gym/openvla-7b+bridge+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--real-world--backdoor_model--kaust--24000_chkpt",
        use_l1_regression = True,
        use_diffusion = False,
        use_film = False,
        num_images_in_input = 2,
        use_proprio = True,
        load_in_8bit = False,
        load_in_4bit = False,
        center_crop = True,
        num_open_loop_steps = NUM_ACTIONS_CHUNK,
        unnorm_key = FLAGS.dataset_stats,
    )

    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)


    # # format the prompt
    prompt = f"In: What action should the robot take to {FLAGS.text_cond}?\nOut:"
    print(prompt)

    # running rollouts
    try:
        for _ in range(100):

            obs, info = env.reset()
            episode_return = 0.0

            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            save_dir = f"frames_eggplant_backdoor_w_trigger"
            frame_saver = FrameSaver(save_dir)

            for i in range(100):
                start_time = time.time()

                img_primary = cv2.cvtColor(obs["image_primary"], cv2.COLOR_RGB2BGR)
                img_wrist = cv2.cvtColor(obs["image_wrist"], cv2.COLOR_RGB2BGR)

                if "state" not in obs:
                    obs["state"] = np.concatenate(
                        [interface.eef_pose[:6], [0.0], [interface.gripper_state]],  # padding
                        dtype=np.float32,
                    )

                if FLAGS.show_img:
                    cv2.imshow("img_primary", img_primary)
                    cv2.imshow("img_wrist", img_wrist)
                    # capture "r" key and reset
                    if cv2.waitKey(10) & 0xFF == ord("r"):
                        break
                
                obs["full_image"] = img_primary
                # obs["image_wrist"] = Image.fromarray(img_wrist)

               

                # save each frame
                
                
                # # single view
                # # frame_saver.save(img_primary)

                # dual view
                frame = np.hstack((img_primary, img_wrist))
                frame_saver.save(frame)

                # Predict Action (7-DoF; un-normalize for BridgeData V2)
                if len(action_queue) == 0:
                    actions= get_vla_action(cfg, vla, processor, obs, FLAGS.text_cond, action_head, proprio_projector)
                    action_queue.extend(actions)
                
                action = action_queue.popleft()
                
                assert (
                    len(action) == 7
                ), f"Action size should be in x, y, z, rx, ry, rz, gripper"


                print("--- VLA inference took %s seconds ---" % (time.time() - start_time))
                print(f" Step {i}: performing action: {action}")

                if FLAGS.clip_actions:
                    action[:6] = np.clip(action[:6], -0.02, 0.02)
                    print(f"Clipped action: {action}")

                # step env -- info contains full "chunk" of observations for logging
                # obs only contains observation for final step of chunk
                obs, reward, done, trunc, info = env.step(action)
                episode_return += reward

                if done:
                    break

            print(f"Episode return: {episode_return}")

    except KeyboardInterrupt:
        env.reset()
        quit()


if __name__ == "__main__":
    app.run(main)
