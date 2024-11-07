import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

from agents.DuelingDQNAgent import DuelingDQNAgent
from common.atari_wrappers import make_atari, wrap_deepmind
from utils import get_device, plot_results

def train(args):
    device = get_device(args.device)
    print("Using device: ", device)

    env = make_atari(args.env_name, render_mode=args.render_mode)
    env = wrap_deepmind(env)

    if args.architecture == 'dueling':
        agent = DuelingDQNAgent(learning_rate=args.learning_rate, n_actions=env.action_space.n, 
                            input_dims=env.observation_space.shape, gamma=args.gamma,
                            epsilon=args.epsilon, min_epsilon=args.min_epsilon, dec_epsilon=args.dec_epsilon, 
                            batch_size=args.batch_size, memory_size=args.memory_size, 
                            replace_network_count=args.replace_network_count, device=device,
                            load_checkpoint_dir=args.load_checkpoint, save_checkpoint_dir=args.save_checkpoint,
                            buffer_type=args.buffer_type, alpha=args.alpha, beta=args.beta)
    
    if args.load_checkpoint:
        agent.load_model()
        print(f"Loaded model from {args.load_checkpoint}")

    scores, epsilon_history, steps = [], [], []
    step_count, best_score = 0, -np.inf

    for i in range(args.n_games):
        obs = env.reset()
        score, done = 0, False

        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward

            agent.store_experience(obs, action, reward, new_obs, int(done))
            agent.learn()

            obs = new_obs
            step_count += 1
        
        scores.append(score)
        epsilon_history.append(agent.epsilon)
        steps.append(step_count)
        avg_score = np.mean(scores)

        if score > avg_score:
            agent.save_model()
        if score > best_score:
            best_score = score

        print(f'Episode {i+1}/{args.n_games} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | '
              f'Best Score: {best_score:.2f} | Epsilon: {agent.epsilon:.3f} | Steps: {step_count}')

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        plot_path = os.path.join(args.plot_dir, f"{args.env_name}_{args.architecture}_{args.n_games}.png")
        plot_results(steps, epsilon_history, scores, plot_path)
        print(f"Training results plotted to {plot_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent with various architectures.")
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the Atari environment')
    parser.add_argument('--n_games', type=int, default=1000, help="Number of games to train")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the agent")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument('--dec_epsilon', type=float, default=0.00001, help="Decrement epsilon for exploration")
    parser.add_argument('--min_epsilon', type=float, default=0.02, help="Minimum epsilon for exploration")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--memory_size', type=int, default=1000, help="Replay buffer size")
    parser.add_argument('--replace_network_count', type=int, default=1000, help="Network replacement frequency")
    parser.add_argument('--architecture', type=str, choices=['dueling', 'dueling double'], default='dueling',
                        help="Choose DQN architecture: dueling or double")
    parser.add_argument('--buffer_type', type=str, choices=['uniform', 'prioritized'], default='uniform',
                        help="Choose replay buffer type: uniform or prioritized")
    parser.add_argument('--alpha', type=float, default=0.6, help="Alpha parameter for PER (prioritization level)")
    parser.add_argument('--beta', type=float, default=0.4, help="Beta parameter for PER (importance-sampling level)")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--save_checkpoint', type=str, default=None, help="Path to save model checkpoint")
    parser.add_argument('--plot_dir', type=str, default=None, help="Plot the training result")
    parser.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='rgb_array',
                        help="Choose render mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
