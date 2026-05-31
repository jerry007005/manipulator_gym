import tensorflow_datasets as tfds

builder = tfds.builder_from_directory(
    "dataset_backdoor--zirun/pick_up_the_carrot_and_place_it_on_the_plate_-1"
)
ds = builder.as_dataset(split="train")

total_steps = 0
num_episodes = 0

for episode in ds:
    steps = list(episode["steps"])

    # 和你视频代码保持一致（去掉第一个 timestep）
    steps = steps[1:]

    num_steps = len(steps)

    total_steps += num_steps
    num_episodes += 1

avg_steps = total_steps / num_episodes

print(f"Total episodes: {num_episodes}")
print(f"Total steps: {total_steps}")
print(f"Average steps per episode: {avg_steps:.2f}")