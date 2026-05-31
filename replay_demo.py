# import tensorflow_datasets as tfds
# import imageio
# import os

# builder = tfds.builder_from_directory("dataset/position_2/lift_eggplant/lift_eggplant_1")
# ds = builder.as_dataset(split="train")

# output_dir = "videos"
# os.makedirs(output_dir, exist_ok=True)

# for ep_idx, episode in enumerate(ds):

#     video_path = f"{output_dir}/episode_{ep_idx:05d}.mp4"

#     # 取出所有steps并去掉第一个 timestep
#     steps = list(episode["steps"])
#     steps = steps[1:]   # 对齐 bridge_orig_dataset_transform

#     with imageio.get_writer(video_path, fps=20) as writer:
#         for step in steps:
#             img = step["observation"]["image_primary"].numpy()
#             writer.append_data(img)

#     print("saved:", video_path)

import tensorflow_datasets as tfds
import imageio
import os

# 数据集路径
data_dir = "dataset--zirun/pick_up_the_eggplant_and_place_it_on_the_plate_2"

# 输出路径
output_path = "last_wrist_frame.png"

# 读取 dataset
builder = tfds.builder_from_directory(data_dir)
ds = builder.as_dataset(split="train")

# 取第一个 episode
first_episode = next(iter(ds))

# 所有 steps
steps = list(first_episode["steps"])

# ⚠️ 和你之前一样，对齐（去掉第一个）
steps = steps[1:]

# 取最后一个 step
last_step = steps[-5]

# 提取 wrist camera（⚠️ 这里名字可能要确认）
img = last_step["observation"]["image_wrist"].numpy()

# 保存图片
imageio.imwrite(output_path, img)

print("Saved:", output_path)