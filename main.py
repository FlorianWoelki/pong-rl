import argparse
import gymnasium as gym
from trainer import PongTrainer
from util import display_frames_as_gif


def main(args: argparse.Namespace):
    render_mode = None
    if args.mode == "render":
        render_mode = "human"
    elif args.mode == "animation":
        render_mode = "rgb_array"

    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    trainer = PongTrainer(env, args.mode)

    if args.mode == "render":
        trainer.play()
    elif args.mode == "animation":
        trainer.play(1000)
        frames = trainer.get_frames()
        display_frames_as_gif(frames)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        help="The mode to run the environment in. Can be 'render' or 'animation'.",
    )
    args = parser.parse_args()
    main(args)
