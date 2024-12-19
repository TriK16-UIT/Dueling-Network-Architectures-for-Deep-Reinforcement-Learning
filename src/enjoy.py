import torch
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from agents.DuelingDQNAgent import DuelingDQNAgent
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from agents.DQNAgent import DQNAgent
from agents.DDQNAgent import DDQNAgent
from common.atari_wrappers import make_atari, wrap_deepmind
from utils import get_device, set_global_seeds

def enjoy(args):
    print("Enjoy parameters:")
    print("(" + "; ".join([f'"{arg}": {value}' for arg, value in vars(args).items()]) + ")")
    print("\nStarting enjoying...\n")

    device = get_device(args.device)
    print("Using device: ", device)

    env = make_atari(args.env_name, enjoy=True, run_name=None, capture_video=False, video_frequency=0)
    # env = gym.make(args.env_name, render_mode="human")
    env = wrap_deepmind(env, episode_life=True, clip_rewards=False)

    #Epsilon = 0.01 to avoid getting stuck
    if args.architecture == 'dueling':
        agent = DuelingDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.05)
    elif args.architecture == 'dueling_double':
        agent = DuelingDDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.05)
    elif args.architecture == 'natural':
        agent = DQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.05)
    elif args.architecture == 'double':
        agent = DDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.05)

    if args.load_checkpoint:
        agent.load_model(args.load_checkpoint)
        print(f"Loaded model from {args.load_checkpoint}")

    agent.q_eval.eval()

    episode = 0
    while True:
        obs = env.reset()
        done = False

        while not done:
            action = agent.choose_action(obs, env)
            new_obs, _, done, info = env.step(action)

            obs = new_obs

        if "episode" in info:
            episode += 1
            print(f"eval_episode={episode}, episodic_return={info['episode']['r']}")


def parse_args():
    """
    Parse command-line arguments for training the DQN agent.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train DQN agent with various architectures.")
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the Atari environment')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--architecture', type=str, choices=['dueling', 'dueling_double', 'natural', 'double'], default='dueling',
                        help="Choose DQN architecture")
    parser.add_argument('--load_checkpoint', type=str, default='tmp/best_model.pth', help="Path to load model checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enjoy(args)