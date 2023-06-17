import torch
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_durations(episode_rewards: list[float], episode_start: int = 0) -> None:
    """
    Plot the episode reward over time.

    Args:
        episode_start: Starting episode index.
    """
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    x_range = [i for i in range(episode_start, len(rewards_t) + episode_start)]
    plt.plot(
        x_range,
        rewards_t.numpy(),
        label="Episode Reward",
    )
    length = 10
    if len(rewards_t) >= length:
        means = rewards_t.unfold(0, length, 1).mean(1).view(-1)
        x_values = torch.arange(length - 1, len(rewards_t))
        plt.plot(
            x_values.numpy() + episode_start,
            means.numpy()[: len(x_values)],
            label=f"Mean ({length}) Reward",
        )

    plt.legend()
    plt.pause(0.001)


def display_frames_as_gif(frames: list[torch.Tensor]) -> None:
    """
    Display a list of frames as a gif and saves it to a file.

    Args:
        frames: A list of frames to display.
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=144)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i: int) -> None:
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    plt.close(anim._fig)
    anim.save("animation.gif")
