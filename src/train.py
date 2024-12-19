import time
import numpy as np
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

from agents.DuelingDQNAgent import DuelingDQNAgent
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from agents.DQNAgent import DQNAgent
from agents.DDQNAgent import DDQNAgent
from common.atari_wrappers import make_atari, wrap_deepmind
from utils import get_device, plot_results, set_global_seeds, generate_seed
from torch.utils.tensorboard import SummaryWriter

def train(args):
    """
    Train the Dueling DQN agent on the specified Atari environment.

    Args:
        args (argparse.Namespace): Command-line arguments specifying training parameters.
    """
    args.seed = generate_seed(args.seed)
    set_global_seeds(args.seed)

    run_name = f"{args.env_name}__{args.architecture}__{args.seed}__{int(time.time())}"

    if args.save_checkpoint is None:
        args.save_checkpoint = f"models/{run_name}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print("Training parameters:")
    print("(" + "; ".join([f'"{arg}": {value}' for arg, value in vars(args).items()]) + ")")
    print("\nStarting training...\n")

    device = get_device(args.device)
    print("Using device: ", device)

    env = make_atari(args.env_name, run_name, args.capture_video, args.video_frequency)
    # test scale=True    
    env = wrap_deepmind(env)
    env.action_space.seed(args.seed)

    if args.architecture == 'dueling':
        agent = DuelingDQNAgent(learning_rate=args.learning_rate, n_actions=env.action_space.n, 
                            input_dims=env.observation_space.shape, gamma=args.gamma,
                            epsilon=args.epsilon, min_epsilon=args.min_epsilon, dec_epsilon=args.dec_epsilon, 
                            batch_size=args.batch_size, memory_size=args.memory_size, device=device,
                            buffer_type=args.buffer_type, clip_grad_norm=args.clip_grad_norm,
                            alpha=args.alpha, beta=args.beta, max_beta=args.max_beta, inc_beta=args.inc_beta)
    elif args.architecture == 'dueling_double':
        agent = DuelingDDQNAgent(learning_rate=args.learning_rate, n_actions=env.action_space.n, 
                            input_dims=env.observation_space.shape, gamma=args.gamma,
                            epsilon=args.epsilon, min_epsilon=args.min_epsilon, dec_epsilon=args.dec_epsilon, 
                            batch_size=args.batch_size, memory_size=args.memory_size, device=device,
                            buffer_type=args.buffer_type, clip_grad_norm=args.clip_grad_norm,
                            alpha=args.alpha, beta=args.beta, max_beta=args.max_beta, inc_beta=args.inc_beta) 
    elif args.architecture == 'natural':
        agent = DQNAgent(learning_rate=args.learning_rate, n_actions=env.action_space.n, 
                            input_dims=env.observation_space.shape, gamma=args.gamma,
                            epsilon=args.epsilon, min_epsilon=args.min_epsilon, dec_epsilon=args.dec_epsilon, 
                            batch_size=args.batch_size, memory_size=args.memory_size, device=device,
                            buffer_type=args.buffer_type, clip_grad_norm=args.clip_grad_norm,
                            alpha=args.alpha, beta=args.beta, max_beta=args.max_beta, inc_beta=args.inc_beta) 
    elif args.architecture == 'double':
        agent = DDQNAgent(learning_rate=args.learning_rate, n_actions=env.action_space.n, 
                            input_dims=env.observation_space.shape, gamma=args.gamma,
                            epsilon=args.epsilon, min_epsilon=args.min_epsilon, dec_epsilon=args.dec_epsilon, 
                            batch_size=args.batch_size, memory_size=args.memory_size, device=device,
                            buffer_type=args.buffer_type, clip_grad_norm=args.clip_grad_norm,
                            alpha=args.alpha, beta=args.beta, max_beta=args.max_beta, inc_beta=args.inc_beta) 


    if args.load_checkpoint:
        agent.load_model(args.load_checkpoint)
        print(f"Loaded model from {args.load_checkpoint}")

    step_count, episode, best_score = 0, 0, -np.inf
    start_time = time.time()
    while step_count <= args.total_timesteps:
        obs = env.reset(seed=args.seed)
        done = False

        while not done:
            agent.decrement_epsilon()
            action = agent.choose_action(obs, env)
            new_obs, reward, done, info = env.step(action)

            agent.store_experience(obs, action, reward, new_obs, done)

            if step_count > args.learning_start and step_count % args.train_frequency == 0:
                q_values, loss = agent.learn()

                if step_count % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, step_count)
                    writer.add_scalar("losses/q_values", q_values, step_count)
                    writer.add_scalar("charts/SPS", int(step_count / (time.time() - start_time)), step_count)

            if step_count > args.learning_start and step_count % args.replace_network_count == 0:
                agent.replace_target_network()

            obs = new_obs
            step_count+=1
        
        # Episode only counts when running out of lives!!! If we place episode+=1 outside this if statement, it goes wrong.
        if "episode" in info:
            episode += 1
            score, length = info["episode"]["r"], info["episode"]["l"]
            writer.add_scalar("charts/episodic_return", score, step_count)
            writer.add_scalar("charts/episodic_length", length, step_count)

            print(f'Episode {episode} | Score: {score:.2f} | '
              f'Best Score: {best_score:.2f} | Epsilon: {agent.epsilon:.3f} | Beta: {agent.beta:.3f} | Steps: {step_count}')
            
             # Save the model if this is the best score achieved
            if score > best_score and step_count > args.learning_start:
                best_score = score
                agent.save_model(args.save_checkpoint, 'best_model.pth')
                print(f"New best score: {best_score:.2f}!")
        
    agent.save_model(args.save_checkpoint, 'last_model.pth')

    env.close()
    writer.close()

def parse_args():
    """
    Parse command-line arguments for training the DQN agent.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train DQN agent with various architectures.")
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4', help='Name of the Atari environment')
    parser.add_argument('--total_timesteps', type=int, default=10000000, help="Number of total timesteps")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the agent")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument('--dec_epsilon', type=float, default=0.00001, help="Decrement epsilon for exploration")
    parser.add_argument('--min_epsilon', type=float, default=0.02, help="Minimum epsilon for exploration")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--memory_size', type=int, default=1000, help="Replay buffer size")
    parser.add_argument('--replace_network_count', type=int, default=1000, help="Network replacement frequency")
    parser.add_argument('--architecture', type=str, choices=['dueling', 'dueling_double', 'natural', 'double'], default='dueling_double',
                        help="Choose DQN architecture")
    parser.add_argument('--buffer_type', type=str, choices=['uniform', 'prioritized'], default='uniform',
                        help="Choose replay buffer type: uniform or prioritized")
    parser.add_argument('--clip_grad_norm', action='store_true', default=False, help="Enable gradient clipping")
    parser.add_argument('--alpha', type=float, default=0.6, help="Alpha parameter for PER (prioritization level)")
    parser.add_argument('--beta', type=float, default=0.4, help="Beta parameter for PER (importance-sampling level)")
    parser.add_argument('--max_beta', type=float, default=1.0, help="Max Beta for PER")
    parser.add_argument('--inc_beta', type=float, default=3e-7, help="Increment beta for PER")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument('--save_checkpoint', type=str, default=None, help="Path to save model checkpoint")
    parser.add_argument('--capture_video', action='store_true', default=False, help="Save training video")
    parser.add_argument('--learning_start', type=int, default=80000, help="Decide when to start training")
    parser.add_argument('--train_frequency', type=int, default=4, help="Training frequency")
    parser.add_argument('--video_frequency', type=int, default=1000, help="Save video after n episodes")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
