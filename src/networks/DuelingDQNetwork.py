import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DuelingDQNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, input_dims, name, save_checkpoint_dir, load_checkpoint_dir, device):
        super(DuelingDQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.Value = nn.Linear(512, 1)
        self.Advantage = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device = device
        self.to(self.device)

        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_checkpoint_name = os.path.join(save_checkpoint_dir, name + ".pt")

        self.load_checkpoint_dir = load_checkpoint_dir
        self.load_checkpoint_name = os.path.join(load_checkpoint_dir, name + ".pt")

    def get_conv_output_dimensions(self, input_dims):
        """Get last convolutional layer output dimenions to feed in Linear layer"""
        temp = torch.zeros(1, *input_dims)
        dim1 = self.conv1(temp)
        dim2 = self.conv2(dim1)
        dim3 = self.conv3(dim2)
        return int(np.prod(dim3.size()))

    
    def forward(self, data):
        """Feed forward the network to get the value, advantage tuple"""
        conv_layer1 = F.relu(self.conv1(data))
        conv_layer2 = F.relu(self.conv2(conv_layer1))
        conv_layer3 = F.relu(self.conv3(conv_layer2))

        output_conv_layer = conv_layer3.view(conv_layer3.size()[0], -1)

        fc_layer1 = F.relu(self.fc1(output_conv_layer))
        fc_layer2 = F.relu(self.fc2(fc_layer1))

        value = self.Value(fc_layer2)
        advantage = self.Advantage(fc_layer2)

        return value, advantage
    
    def save_checkpoint(self):
        """Saves the checkpoint to the desired file"""
        print('Saving checkpoint...')
        os.makedirs(self.save_checkpoint_dir, exist_ok=True)
        print(self.save_checkpoint_name)
        torch.save(self.state_dict(), self.save_checkpoint_name)

    def load_checkpoint(self):
        """Loads the checkpoint from the saved file"""
        print('Loading checkpoint...')
        if not os.path.isfile(self.load_checkpoint_name):
            raise FileNotFoundError(f"Checkpoint file '{self.load_checkpoint_name}' not found.")
        print(self.load_checkpoint_name)
        self.load_state_dict(torch.load(self.load_checkpoint_name))
        print("Checkpoint loaded successfully.")
    


    
