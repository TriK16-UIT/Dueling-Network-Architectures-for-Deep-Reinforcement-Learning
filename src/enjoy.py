import torch
import gym
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from agents.DuelingDQNAgent import DuelingDQNAgent
from common.atari_wrappers import make_atari, wrap_deepmind
from utils import get_device, set_global_seeds

def enjoy(args):
    print(args.seed)
    set_global_seeds(args.seed)

    print("Enjoy parameters:")
    print("(" + "; ".join([f'"{arg}": {value}' for arg, value in vars(args).items()]) + ")")
    print("\nStarting enjoying...\n")

    device = get_device(args.device)
    print("Using device: ", device)

    env = make_atari(args.env_name, render_mode="human")
    # env = gym.make(args.env_name, render_mode="human")
    if args.seed is not None:
        env.seed(args.seed)
    env = wrap_deepmind(env, episode_life=True)


    #Epsilon = 0.01 to avoid getting stuck
    if args.architecture == 'dueling':
        agent = DuelingDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.01)

    if args.load_checkpoint:
        agent.load_model(args.load_checkpoint)
        print(f"Loaded model from {args.load_checkpoint}")

    episode_count = 0
    while True:
        obs = env.reset()
        score, done = 0, False

        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward

            obs = new_obs

        episode_count += 1
        print(f"Episode {episode_count} | Score: {score:.2f}")


def parse_args():
    """
    Parse command-line arguments for training the DQN agent.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train DQN agent with various architectures.")
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the Atari environment')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--architecture', type=str, choices=['dueling', 'dueling double'], default='dueling',
                        help="Choose DQN architecture: dueling or double")
    parser.add_argument('--load_checkpoint', type=str, default='tmp/best_model.pth', help="Path to load model checkpoint")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enjoy(args)