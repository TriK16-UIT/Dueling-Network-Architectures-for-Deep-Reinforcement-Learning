import torch
import warnings
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

def get_device(device):
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        warnings.warn("CUDA not available. Falling back to CPU.")
        return torch.device('cpu')

def plot_results(steps, epsilon_history, scores, plot_path):
    figure = plt.figure()
    plot1 = figure.add_subplot(1, 1, 1, label='plot1')
    plot2 = figure.add_subplot(1, 1, 1, label='plot2')

    plot1.plot(steps, epsilon_history, color='C0')
    plot1.set_xlabel('No. of steps', color='C0')
    plot1.set_ylabel('Epsilon', color='C0')
    plot1.tick_params(axis='x', color='C0')
    plot1.tick_params(axis='y', color='C0')
    
    running_avg = np.empty(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i - 30): i + 1])

    plot2.plot(steps, scores, color='C1')
    plot2.axes.get_xaxis().set_visible(False)
    plot2.yaxis.tick_right()
    plot2.set_ylabel('Avg. scores', color='C1')
    plot2.yaxis.set_label_position('right')
    plot2.tick_params(axis='y', color='C1')

    plt.tight_layout()
    plt.savefig(plot_path)

def set_global_seeds(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False