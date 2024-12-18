from networks.DuelingDQNetwork import DuelingDQNetwork
from replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import torch
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_

class DuelingDQNAgent(object):
    def __init__(self, n_actions, input_dims, learning_rate=1e-4, gamma=0.99,
                 epsilon=1.0, batch_size=32, memory_size=1000,
                 clip_grad_norm=False, alpha=0.6, beta=0.4, max_beta=1.0, inc_beta=3e-7,
                 dec_epsilon=1e-5, min_epsilon=0.01, device="cpu", buffer_type='uniform'):
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
        clip_grad_norm (bool, optional): Whether to apply gradient clipping. Default is False.
        alpha (float, optional): Prioritization exponent for prioritized replay buffer. Default is 0.6.
        beta (float, optional): Initial value of beta for importance-sampling weights. Default is 0.4.
        max_beta (float, optional): Maximum value of beta during annealing. Default is 1.0.
        inc_beta (float, optional): Inc rate of beta after each step. Default is 3e-7.
        dec_epsilon (float, optional): Decay rate of epsilon after each step. Default is 1e-5.
        min_epsilon (float, optional): Minimum value of epsilon. Default is 0.1.
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
        self.clip_grad_norm = clip_grad_norm
        self.alpha = alpha
        self.beta = beta
        self.max_beta = max_beta
        self.inc_beta = inc_beta
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.action_indices = [i for i in range(n_actions)]
        self.device = device
        self.buffer_type = buffer_type
        
        self.q_eval = DuelingDQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=self.input_dims, device=self.device)
        self.q_next = DuelingDQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=self.input_dims, device=self.device)
        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()
        
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
        self.q_next.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, observation, env):
        """
        Chooses an action using the epsilon-greedy policy.

        Args:
            observation (numpy.ndarray): The current state observation.

        Returns:
            int: The chosen action.
        """
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
                q_values = self.q_eval.forward(state)
                action = torch.argmax(q_values).item()
        else:
            action = env.action_space.sample()

        return action
    
    def learn(self):
        """
        Train the agent on a batch of experiences.
        """
        self.q_eval.optimizer.zero_grad()

        state, action, reward, next_state, done, weights, indices = self.get_sample_experience()
        batch_indices = torch.arange(self.batch_size, device=self.device)

        q_pred = self.q_eval.forward(state)
        q_pred = q_pred[batch_indices, action]
        
        with torch.no_grad():
            q_next = self.q_next.forward(next_state)
            q_next_max = q_next.max(dim=1)[0]
            q_next_max[done] = 0.0
            
        q_target = reward + self.gamma * q_next_max
        q_target[done] = reward[done]

        # Compute loss
        loss = self.q_eval.loss(q_pred, q_target).to(self.device)
        loss = (weights * loss).mean()
        loss.backward()

        #Gradient clipping (10.0 based on paper)
        if self.clip_grad_norm:
            clip_grad_norm_(self.q_eval.parameters(), 10.0)

        self.q_eval.optimizer.step()
        if self.buffer_type == 'prioritized':
            self.increment_beta()

        if self.buffer_type == 'prioritized' and indices is not None:
            td_errors = torch.abs(q_target - q_pred).detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, td_errors)

        return q_pred.mean().item(), loss

    def save_model(self, directory, filename):
        """
        Saves the model's state_dict to the specified directory with the given filename.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        self.q_eval.save_checkpoint(path)
        print(f"Model saved as {filename} in {directory}")

    def load_model(self, path):
        """
        Loads the model's state_dict from the specified path.
        """
        self.q_eval.load_checkpoint(path)
        # Optionally, you can also load the state_dict into q_next if needed
        self.q_next.load_state_dict(self.q_eval.state_dict())
        print(f"Model loaded from {path}")

