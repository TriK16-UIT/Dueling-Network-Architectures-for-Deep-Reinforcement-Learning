from networks.DuelingDQNetwork import DuelingDQNetwork
from replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import torch
import numpy as np
import os

class DuelingDQNAgent(object):
    def __init__(self, learning_rate, n_actions, input_dims, gamma,
                 epsilon, batch_size, memory_size, replace_network_count, alpha=0.6, beta=0.4,
                 dec_epsilon=1e-5, min_epsilon=0.1, save_checkpoint_dir=None, load_checkpoint_dir=None, device="cpu", buffer_type='uniform'):
        """
        Initialize Dueling DQN Agent with improved type hints and documentation.
        
        Args:
            learning_rate: Learning rate for optimizer
            n_actions: Number of possible actions
            input_dims: Dimensions of input state
            gamma: Discount factor
            epsilon: Initial exploration rate
            batch_size: Size of training batch
            memory_size: Size of replay buffer
            replace_network_count: Steps before target network update
            dec_epsilon: Epsilon decay rate
            min_epsilon: Minimum exploration rate
            checkpoint_dir: Directory to save model checkpoints
            device: Computing device (cpu/cuda)
        """
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_network_count = replace_network_count
        self.alpha = alpha
        self.beta = beta
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
        Choose action using epsilon-greedy policy.
        
        Args:
            observation: Current state observation
            
        Returns:
            Selected action index
        """
        self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon

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
            torch.tensor(done).to(self.device),
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

                action = torch.argmax(advantages).item()
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
        q_next[done] = 0.0
        q_target = reward + self.gamma * q_next.max(dim=1)[0]

        #MSELoss
        # loss = (weights * (q_target - q_pred) ** 2).mean()
        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        loss = (weights * loss).mean()
        loss.backward()

        self.q_eval.optimizer.step()
        self.decrement_epsilon()
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

