import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 
# 1. DQNMLP: change the shape of the output layer from num_actions to 1.
# 2. DQNCNN: figure out how to incorporate action into model?
class DQNMLP(nn.Module):
    def __init__(self, input_size, num_actions, hidden_layers_sizes):
        """
        Initialize the MLP for Deep Q-Learning.

        :param input_size: The size of the input (flattened state size).
        :param num_actions: The number of possible actions.
        :param hidden_layers_sizes: A list containing the number of units in each hidden layer.
        
        """
        super(DQNMLP, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        
        # First hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers_sizes[0]))
        
        # Remaining hidden layers
        for i in range(1, len(hidden_layers_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers_sizes[-1], num_actions)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

class DQNCNN(nn.Module):
    '''
    CNN based agent for deep Q learning.
    TODO: check the conv_output_size.
    '''
    def __init__(self, input_channels, num_actions, hidden_size):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the input to the first fully connected layer
        # This assumes the input height and width is 8x8. Adjust accordingly if different.
        conv_output_size = 64 * 8 * 8 ### need to look into this!
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

