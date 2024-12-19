import time
import torch
import gym
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from agents.DuelingDQNAgent import DuelingDQNAgent
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from agents.DQNAgent import DQNAgent
from agents.DDQNAgent import DDQNAgent
from utils import get_device, plot_results, set_global_seeds, generate_seed
from common.atari_wrappers import make_atari, wrap_deepmind



def evaluate(args):
    args.seed = generate_seed(args.seed)
    set_global_seeds(args.seed)

    run_name = f"{args.env_name}__{args.architecture}__{args.seed}__{int(time.time())}-eval"


    print("Evaluating parameters:")
    print("(" + "; ".join([f'"{arg}": {value}' for arg, value in vars(args).items()]) + ")")
    print("\nStarting evaluating...\n")

    device = get_device(args.device)
    print("Using device: ", device)

    env = make_atari(args.env_name, run_name, args.capture_video, args.video_frequency)
    env = wrap_deepmind(env, episode_life=False)
    env.action_space.seed(args.seed)

    if args.architecture == 'dueling':
        agent = DuelingDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.01)
    elif args.architecture == 'dueling_double':
        agent = DuelingDDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.01)
    elif args.architecture == 'natural':
        agent = DQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.01)
    elif args.architecture == 'double':
        agent = DDQNAgent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, device=args.device, epsilon=0.01)

    if args.load_checkpoint:
        agent.load_model(args.load_checkpoint)
        print(f"Loaded model from {args.load_checkpoint}")

    agent.q_eval.eval()

    episodic_returns = []
    episode = 0
    while episode < args.n_games:
        obs = env.reset(seed=args.seed)
        done = False

        while not done:
            action = agent.choose_action(obs, env)
            new_obs, _, done, info = env.step(action)

            obs = new_obs

        if "episode" in info:
            episode+=1
            print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            episodic_returns += [info["episode"]["r"]]

    env.close()

def parse_args():
    """
    Parse command-line arguments for training the DQN agent.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train DQN agent with various architectures.")
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the Atari environment')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--n_games', type=int, default=10, help="Number of games for evaluating")
    parser.add_argument('--architecture', type=str, choices=['dueling', 'dueling_double', 'natural', 'double'], default='dueling_double',
                        help="Choose DQN architecture")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--capture_video', action='store_true', default=False, help="Save training video")
    parser.add_argument('--video_frequency', type=int, default=1, help="Save video after n episodes")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

    