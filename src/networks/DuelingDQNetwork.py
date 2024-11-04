import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DuelingDQNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, input_dims, name, checkpoint_dir, device):
        super(DuelingDQNetwork, self).__init__()
        """According to paper, recommended input dims are 4, 32, 32"""
        self.net = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device = device
        self.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(self.checkpoint_dir, name)

    def get_conv_output_dimensions(self, input_dims):
        """Get last convolutional layer output dimenions to feed in Linear layer"""
        dummy_input = torch.zeros(1, *input_dims).to(self.device)
        output = self.net(dummy_input)
        conv_output_dims = output.view(output.size(0), -1).size(1)
        return conv_output_dims
    
    def forward(self, data):
        """Feed forward the network to get the value, advantage tuple"""
        conv_out = self.net(data)
        flat = conv_out.view(conv_out.size()[0], -1)

        fc1_out = F.relu(self.fc1(flat))
        fc2_out = F.relu(self.fc2(fc1_out))

        value = self.value(fc2_out)
        advantage = self.advantage(fc2_out)

        return value, advantage
    
    def save_checkpoint(self):
        """Saves the checkpoint to the desired file"""
        print('Saving checkpoint...')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        """Loads the checkpoint from the saved file"""
        print('Loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_name))

    


    
