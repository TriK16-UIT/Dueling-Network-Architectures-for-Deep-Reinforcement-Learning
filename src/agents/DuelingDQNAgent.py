from networks.DuelingDQNetwork import DuelingDQNetwork
from replay_buffers.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import torch
import numpy as np

class DuelingDQNAgent(object):
    def __init__(self, learning_rate, n_actions, input_dims, gamma,
                 epsilon, batch_size, memory_size, replace_network_count,
                 dec_epsilon=1e-5, min_epsilon=0.1, checkpoint_dir='/models/ddqn/'):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replace_network_count = replace_network_count
        self.dec_epsilon = dec_epsilon
        self.min_epsilon = min_epsilon
        self.checkpoint_dir = checkpoint_dir
        self.action_indices = [i for i in range(n_actions)]
        self.learn_steps_count = 0
        
        self.q_eval = DuelingDQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=self.input_dims, name='q_eval',
                                       checkpoint_dir=self.checkpoint_dir)
        self.q_next = DuelingDQNetwork(learning_rate=self.learning_rate, n_actions=self.n_actions,
                                       input_dims=self.input_dims, name='q_next',
                                       checkpoint_dir=self.checkpoint_dir)
        self.replay_buffer = ReplayBuffer(memory_size=self.memory_size)

    def decrement_epsilon(self):
        """
        Decrements the epsilon after each step till it reaches minimum epsilon (0.1)
        epsilon = epsilon - decrement (default is 0.99e-6)
        """
        self.epsilon = self.epsilon - self.dec_epsilon if self.epsilon > self.min_epsilon else self.min_epsilon

    def store_experience(self, state, action, reward, next_state, done):
        """
        Saves the experience to the replay memory
        """
        self.replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def get_sample_experience(self):
        """
        Returns a sample experience from the replay buffer for learning purpose
        """
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size=self.batch_size)

        t_state = torch.tensor(state).to(self.q_eval.device)
        t_action = torch.tensor(action).to(self.q_eval.device)
        t_reward = torch.tensor(reward).to(self.q_eval.device)
        t_next_state = torch.tensor(next_state).to(self.q_eval.device)
        t_done = torch.tensor(done).to(self.q_eval.device)

        return t_state, t_action, t_reward, t_next_state, t_done
    
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
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            value, advantages = self.q_eval.forward(state)

            action = torch.argmax(advantages).item()
        else:
            action = np.random.choice(self.n_actions)

        return action
    
    def learn(self):
        if self.replay_buffer._next_idx < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        state, action, reward, next_state, done = self.get_sample_experience()
        batches = np.arange(self.batch_size)

        value_s, advantage_s = self.q_eval.forward(state)
        value_s_dash, advantage_s_dash = self.q_next.forward(next_state)

        q_pred = torch.add(value_s, advantage_s - advantage_s.mean(dim=1, keepdim=True))[batches, action]
        q_next = torch.add(value_s_dash, advantage_s_dash - advantage_s_dash.mean(dim=1, keepdim=True))
        max_q_next = q_next.max(dim=1)

        q_next[done] = 0.0
        q_target = reward + self.gamma * max_q_next[0]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.decrement_epsilon()
        self.learn_steps_count += 1

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

