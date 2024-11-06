import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from agents.DuelingDQNAgent import DuelingDQNAgent
from common.atari_wrappers import make_atari, wrap_deepmind

if __name__ == "__main__":
    n_games = 1000
    scores = []
    epsilon_history = []
    steps = []
    step_count = 0
    best_score = -np.inf
    load_checkpoint = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    env = make_atari("PongNoFrameskip-v4")
    env = wrap_deepmind(env)

    agent = DuelingDQNAgent(learning_rate=1e-4, n_actions=env.action_space.n, 
                            input_dims=env.observation_space.shape, gamma=0.99,
                            epsilon=1.0, min_epsilon=0.02, batch_size=32, memory_size=10000,
                            replace_network_count=1000, device=device)
    
    for i in range(1000):
        obs = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.store_experience(obs, action, reward, new_obs, int(done))
                agent.learn()
            obs = new_obs
            step_count += 1
        scores.append(score)
        epsilon_history.append(agent.epsilon)
        steps.append(step_count)
        avg_score = np.mean(scores)

        if score > avg_score:
            if not load_checkpoint:
                agent.save_model()

        if score > best_score:
            best_score = score

        print('episode: ', i, ' score: ', score, ' avg. score: ', avg_score,
              ' best_score: ', best_score, ' epsilon: ', agent.epsilon, ' steps ', step_count)
        
    figure = plt.figure()
    plot1 = figure.add_subplot(1, 1, 1, label='plot1')
    plot2 = figure.add_subplot(1, 1, 1, label='plot2')

    plot1.plot(steps, epsilon_history, color='C0')
    plot1.set_xlabel('No. of steps', color='C0')
    plot1.set_ylabel('Epsilon', color='C0')
    plot1.tick_params(axis='x', color='C0')
    plot1.tick_params(axis='y', color='C0')

    # Taking avg. of last 30 scores to avoid fluctuations
    running_avg = np.empty(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i - 30): i + 1])

    plot2.plot(steps, scores, color='C1')
    plot2.axes.get_xaxis().set_visible(False)
    plot2.yaxis.tick_right()
    plot2.set_ylabel('Avg. scores', color='C1')
    plot2.yaxis.set_label_position('right')
    plot2.tick_params(axis='y', color='C1')

    plot_file_name = 'duelingdqn_results.png'
    plot_dir = os.path.join(os.getcwd(), 'plots/')

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('haha')

    plt.savefig(plot_dir + plot_file_name)

        

