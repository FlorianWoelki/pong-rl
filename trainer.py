import os
import time
import datetime
import torch
import gymnasium as gym
import random

from model import PolicyNetwork
from util import plot_durations

UP = 2
DOWN = 3


class PongTrainer:
    def __init__(self, env: gym.Env, mode: str = None) -> None:
        self.device = torch.device("cpu")
        self.mode = mode
        self.episode_rewards: list[float] = []
        self.env = env

        if mode == "animation":
            self.frames: list[torch.Tensor] = []

        # Input dimensionality: 80x80 grid.
        self.input_size = 80 * 80
        self.hidden_size = 200
        self.learning_rate = 7e-4
        # Discount factor for reward.
        self.discount_factor = 0.99

        self.batch_size = 4
        self.save_every_batches = 5

        self.model = PolicyNetwork(self.device, self.input_size, self.hidden_size)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses a 210x160x3 uint8 frame into a 6400 (80x80) 1D float vector.

        Args:
            image: The input image.

        Returns:
            The preprocessed image.
        """
        image = torch.Tensor(image).to(self.device)
        # Crops the image.
        image = image[35:195]
        # Downsample by factor of 2.
        image = image[::2, ::2, 0]
        # Erase background (background type 1).
        image[image == 144] = 0
        # Erase background (background type 2).
        image[image == 109] = 0
        # Set everything else (paddles, ball) to 1.
        image[image != 0] = 1
        return image.flatten().float()

    def calc_discounted_future_rewards(self, rewards: list[float]) -> torch.Tensor:
        """
        Calculate the discounted future rewards.

        Args:
            rewards: List of rewards for each timestep.
            discount_factor: Discount factor for future rewards.

        Returns:
            discounted_future_rewards: Tensor of discounted future rewards.
        """
        discounted_future_rewards = torch.empty(len(rewards)).to(self.device)

        # Compute `discounted_future_reward` for each timestep by iterating backwards
        # from end of episode to beginning because the future reward for a given timestep
        # depends on the rewards of all the timesteps that come after it.
        discounted_future_reward = 0
        for timestep in range(len(rewards) - 1, -1, -1):
            # If `rewards[t] != 0`, we are at game boundary (win or loss) so we # reset
            # `discounted_future_reward` to `0` (this is pong specific!).
            if rewards[timestep] != 0:
                discounted_future_reward = 0

            # Calculate discounted reward based on definition of discounted reward.
            discounted_future_reward = (
                rewards[timestep] + self.discount_factor * discounted_future_reward
            )
            discounted_future_rewards[timestep] = discounted_future_reward

        return discounted_future_rewards

    def load_model(self, model_path: str) -> tuple[str, int]:
        """
        Load the model from a checkpoint.

        Args:
            model_path: The path to the model checkpoint.

        Returns:
            start_time: The start time of the model.
            last_batch: The last batch that was trained.
        """
        start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
        last_batch = -1
        if os.path.exists(model_path):
            print("Loading from checkpoint...")
            save_dict = torch.load(model_path)

            self.model.load_state_dict(save_dict["model_weights"])
            start_time = save_dict["start_time"]
            last_batch = save_dict["last_batch"]
        return start_time, last_batch

    def run_episode(
        self,
    ) -> tuple[torch.Tensor, float]:
        """
        Run a single episode of the environment with the model.

        Returns:
            loss: The loss value for the episode.
            episode_reward: The total reward for the episode
        """
        observation, _ = self.env.reset()
        prev_x = self.preprocess(observation)

        action_chosen_log_probs: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []

        done = False
        timestamp = 0

        while not done:
            cur_x = self.preprocess(observation)
            # Calculate the difference of the current and previous observation.
            x = cur_x - prev_x
            prev_x = cur_x

            # Forward pass through the model to get the probability of moving up.
            prob_up = self.model(x)
            action = UP if random.random() < prob_up else DOWN

            # Calculate the log probability of the action chosen for policy gradient
            # update step.
            action_chosen_prob = prob_up if action == UP else (1 - prob_up)
            action_chosen_log_probs.append(
                torch.log(action_chosen_prob).to(self.device)
            )

            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            rewards.append(torch.Tensor([reward]).to(self.device))
            timestamp += 1

        action_chosen_log_probs = torch.cat(action_chosen_log_probs).to(self.device)
        rewards = torch.cat(rewards).to(self.device)

        discounted_future_rewards = self.calc_discounted_future_rewards(rewards)

        # Normalize rewards to stabilize the training.
        discounted_future_rewards = (
            discounted_future_rewards - discounted_future_rewards.mean()
        ) / discounted_future_rewards.std()

        # Calculate policy loss using the policy gradient loss formula.
        # Measures how much better the action taken was compared to the average action
        # at that state.
        loss = -(discounted_future_rewards * action_chosen_log_probs).sum()
        return loss, rewards.sum()

    def play(self, stop_frame: int = -1) -> None:
        """
        Play the game using the model. The game will be rendered if `mode` is set to
        `render` and will be animated if `mode` is set to `animation`.

        Args:
            stop_frame: The frame to stop at. If set to -1, the game will play until
                        the end.
        """
        self.load_model("checkpoint.pth")
        observation, _ = self.env.reset()
        prev_x = self.preprocess(observation)

        i = 0
        while True:
            time.sleep(1 / 30)
            if self.mode == "animation":
                self.frames.append(self.env.render())
            elif self.mode == "render":
                self.env.render()

            cur_x = self.preprocess(observation)
            x = cur_x - prev_x
            prev_x = cur_x

            prob_up = self.model(x)
            action = UP if random.random() < prob_up else DOWN

            observation, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            if done:
                return

            if self.mode == "animation":
                if i % 10 == 0:
                    print(f"Frame finished: {i}")

                if i == stop_frame:
                    return

                i += 1

    def train(self) -> None:
        """
        Train the policy network model and save the model weights to a checkpoint.
        """
        start_time, last_batch = self.load_model("checkpoint.pth")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        batch = last_batch + 1
        while True:
            mean_batch_loss = 0
            mean_batch_reward = 0
            for _ in range(self.batch_size):
                loss, episode_reward = self.run_episode()
                mean_batch_loss += loss / self.batch_size
                mean_batch_reward += episode_reward / self.batch_size

                self.episode_rewards.append(episode_reward)
                plot_durations(self.episode_rewards, (last_batch + 1) * self.batch_size)
                print(f"Episode reward: {episode_reward}, Episode loss: {loss}")

            optimizer.zero_grad()
            mean_batch_loss.backward()
            optimizer.step()

            print(
                f"Batch: {batch}, mean loss: {mean_batch_loss}, "
                f"mean reward: {mean_batch_reward}"
            )

            if batch != 0 and batch % self.save_every_batches == 0:
                print("Saving checkpoint...")
                save_dict = {
                    "model_weights": self.model.state_dict(),
                    "start_time": start_time,
                    "last_batch": batch,
                }
                torch.save(save_dict, "checkpoint.pth")

            batch += 1

    def get_frames(self) -> list[torch.Tensor]:
        return self.frames
