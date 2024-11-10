import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DuelingDQNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture with convolutional and fully connected layers.

    Args:
        n_actions (int): Number of possible actions.
        input_dims (tuple): Dimensions of the input state.
        name (str): Name of the network (used for saving).
        save_checkpoint_dir (str): Directory to save the model checkpoint.
        load_checkpoint_dir (str): Directory to load the model checkpoint from.
        device (str): Device to run computations on ('cpu' or 'cuda').
    """
    def __init__(self, learning_rate, n_actions, input_dims, device):
        super(DuelingDQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.conv_output_dims = self.get_conv_output_dimensions(input_dims)

        self.fc1 = nn.Linear(self.conv_output_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

        # According to many implementations, Adam and Huber Loss are better compared to RMSprop and MSELoss
        # self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        # self.loss = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.SmoothL1Loss(reduction='none')
        
        self.device = device
        self.to(self.device)

    def get_conv_output_dimensions(self, input_dims):
        """
        Computes the number of output features after the convolutional layers.

        Args:
            input_dims (tuple): Dimensions of the input state.

        Returns:
            int: The total number of features output by the convolutional layers.
        """
        with torch.no_grad():
            temp = torch.zeros(1, *input_dims)
            dim1 = self.conv1(temp)
            dim2 = self.conv2(dim1)
            dim3 = self.conv3(dim2)
        return int(np.prod(dim3.size()))

    
    def forward(self, data):
        """
        Performs a forward pass through the network.

        Args:
            data (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Tensor containing the Q-values for each action.
        """
        conv_layer1 = F.relu(self.conv1(data))
        conv_layer2 = F.relu(self.conv2(conv_layer1))
        conv_layer3 = F.relu(self.conv3(conv_layer2))

        output_conv_layer = conv_layer3.view(conv_layer3.size()[0], -1)

        fc_layer1 = F.relu(self.fc1(output_conv_layer))
        fc_layer2 = F.relu(self.fc2(fc_layer1))

        value = self.value(fc_layer2)
        advantage = self.advantage(fc_layer2)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    
    def save_checkpoint(self, path):
        """
        Saves the model's state_dict to the specified path.
        """
        torch.save(self.state_dict(), path)
    def load_checkpoint(self, path):
        """
        Loads the model's state_dict from the specified path.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file '{path}' not found.")
        self.load_state_dict(torch.load(path, map_location=self.device))
    


    
