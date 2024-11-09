from networks.DuelingDQNetwork import DuelingDQNetwork
from replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import torch
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_

class DuelingDQNAgent(object):
    def __init__(self, learning_rate, n_actions, input_dims, gamma,
                 epsilon, batch_size, memory_size, replace_network_count, clip_grad_norm=False, alpha=0.6, beta=0.4, max_beta=1.0, beta_iters=1e+6,
                 dec_epsilon=1e-5, min_epsilon=0.1, save_checkpoint_dir=None, load_checkpoint_dir=None, device="cpu", buffer_type='uniform'):
        """
        Initialize the Dueling DQN Agent.

        Args:
        learning_rate (float): Learning rate for the optimizer.
        n_actions (int): Number of possible actions.
        input_dims (tuple or int): Dimensions of the input state.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate for epsilon-greedy action selection.
        batch_size (int): Size of the training batch.
        memory_size (int): Maximum size of the replay buffer.
        replace_network_count (int): Number of steps before updating the target network.
        clip_grad_norm (bool, optional): Whether to apply gradient clipping. Default is False.
        alpha (float, optional): Prioritization exponent for prioritized replay buffer. Default is 0.6.
        beta (float, optional): Initial value of beta for importance-sampling weights. Default is 0.4.
        max_beta (float, optional): Maximum value of beta during annealing. Default is 1.0.
        beta_iters (int, optional): Number of iterations over which beta is annealed. Default is 1e6.
        dec_epsilon (float, optional): Decay rate of epsilon after each step. Default is 1e-5.
        min_epsilon (float, optional): Minimum value of epsilon. Default is 0.1.
        save_checkpoint_dir (str, optional): Directory to save model checkpoints. Default is None.
        load_checkpoint_dir (str, optional): Directory to load model checkpoints from. Default is None.
        device (str, optional): Device to run computations on ('cpu' or 'cuda'). Default is 'cpu'.
        buffer_type (str, optional): Type of replay buffer ('uniform' or 'prioritized'). Default is 'uniform'.
        """
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_network_count = replace_network_count
        self.clip_grad_norm = clip_grad_norm
        self.alpha = alpha
        self.beta = beta
        self.max_beta = max_beta
        self.inc_beta = (self.max_beta - self.beta) / beta_iters
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.action_indices = [i for i in range(n_actions)]
        self.learn_steps_count = 0
        self.device = device
        self.buffer_type = buffer_type
        self.save_checkpoint_dir = save_checkpoint_dir or os.path.join(os.getcwd(), 'tmp/ddqn/')
        self.load_checkpoint_dir = load_checkpoint_dir or os.path.join(os.getcwd(), 'tmp/ddqn/')
        
        self.q_eval = DuelingDQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=self.input_dims, name='q_eval',
                                       save_checkpoint_dir=self.save_checkpoint_dir, 
                                       load_checkpoint_dir=self.load_checkpoint_dir, 
                                       device=self.device)
        self.q_next = DuelingDQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=self.input_dims, name='q_next',
                                       save_checkpoint_dir=self.save_checkpoint_dir,
                                       load_checkpoint_dir=self.load_checkpoint_dir,
                                       device=self.device)
        
        if buffer_type == 'uniform':
            self.replay_buffer = ReplayBuffer(memory_size=self.memory_size)
        else:
            self.replay_buffer = PrioritizedReplayBuffer(size=self.memory_size, alpha=self.alpha)

    def decrement_epsilon(self):
        """
        Decay epsilon by a predefined rate until it reaches the minimum value.
        """
        self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon

    def increment_beta(self):
        """
        Increase beta by a predefined rate until it reaches the maximum value.
        """ 
        self.beta = self.beta + self.inc_beta if self.beta < self.max_beta else self.max_beta

    def store_experience(self, state, action, reward, next_state, done):
        """
        Saves the experience to the replay memory
        """
        self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def get_sample_experience(self):
        """
        Sample a batch of experiences from replay buffer.
        
        Returns:
            Tuple of (state, action, reward, next_state, done, weights, indices) tensors
        """
        if self.buffer_type == 'uniform':
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size=self.batch_size)
            weights, indices = np.ones(self.batch_size), None
        else:
            state, action, reward, next_state, done, weights, indices = self.replay_buffer.sample(batch_size=self.batch_size, beta=self.beta)

        return (
            torch.tensor(state, dtype=torch.float32).to(self.device),
            torch.tensor(action).to(self.device),
            torch.tensor(reward, dtype=torch.float32).to(self.device),
            torch.tensor(next_state, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.bool).to(self.device),
            torch.tensor(weights, dtype=torch.float32).to(self.device),
            indices
        )
    
    def replace_target_network(self):
        """
        Updates the parameters after replace_network_count steps
        """
        if self.learn_steps_count % self.replace_network_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, observation):
        """
        Chooses an action with epsilon-greedy method
        """

        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor([observation], dtype=torch.float).to(self.device)
                value, advantages = self.q_eval.forward(state)
                q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
                action = torch.argmax(q_values).item()
        else:
            action = np.random.choice(self.n_actions)

        return action
    
    def learn(self):
        """Train the agent on a batch of experiences."""
        if self.replay_buffer._next_idx < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action, reward, next_state, done, weights, indices = self.get_sample_experience()
        batch_indices = torch.arange(self.batch_size)

        value_s, advantage_s = self.q_eval.forward(state)
        with torch.no_grad():
            value_s_dash, advantage_s_dash = self.q_next.forward(next_state)

        q_pred = value_s + (advantage_s - advantage_s.mean(dim=1, keepdim=True))
        q_pred = q_pred[batch_indices, action]
        
        q_next = value_s_dash + (advantage_s_dash - advantage_s_dash.mean(dim=1, keepdim=True))
        q_next_max = q_next.max(dim=1)[0]
        q_next_max[done] = 0.0
        q_target = reward + self.gamma * q_next_max

        #MSELoss
        # loss = (weights * (q_target - q_pred) ** 2).mean()
        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        loss = (weights * loss).mean()
        loss.backward()

        #Gradient clipping (10.0 based on paper)
        if self.clip_grad_norm:
            clip_grad_norm_(self.q_eval.parameters(), 10.0)

        self.q_eval.optimizer.step()
        self.decrement_epsilon()
        self.increment_beta()
        self.learn_steps_count += 1

        if self.buffer_type == 'prioritized' and indices is not None:
            td_errors = torch.abs(q_target - q_pred).detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, td_errors)

    def save_model(self):
        """
        Saves the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_model(self):
        """
        Loads the values of q_eval and q_next at the checkpoint
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

