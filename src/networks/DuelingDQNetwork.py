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
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1)

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.Value = nn.Linear(512, 1)
        self.Advantage = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device = device
        self.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(self.checkpoint_dir, name)

    def get_conv_output_dimensions(self, input_dims):
        """Get last convolutional layer output dimenions to feed in Linear layer"""
        temp = torch.zeros(1, *input_dims)
        dim1 = self.conv1(temp)
        dim2 = self.conv2(dim1)
        dim3 = self.conv3(dim2)
        return int(np.prod(dim3.size()))
    
    def forward(self, data):
        """Feed forward the network to get the value, advantage tuple"""
        conv1_output = F.relu(self.conv1(data))
        conv2_output = F.relu(self.conv2(conv1_output))
        conv3_output = F.relu(self.conv3(conv2_output))

        conv_output = conv3_output.view(conv3_output.size()[0], -1)

        fc1_output = F.relu(self.fc1(conv_output))
        fc2_output = F.relu(self.fc2(fc1_output))

        value = self.Value(fc2_output)
        advantage = self.Advantage(fc2_output)

        return value, advantage
    
    def save_checkpoint(self):
        """Saves the checkpoint to the desired file"""
        print('Saving checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_name)

    def load_checkpoint(self):
        """Loads the checkpoint from the saved file"""
        print('Loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_name))

    


    
