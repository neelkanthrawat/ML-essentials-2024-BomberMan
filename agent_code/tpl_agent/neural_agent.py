import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import copy

# TODO: 
# 1. DQNMLP: change the shape of the output layer from num_actions to 1.
# 2. DQNCNN: figure out how to incorporate action into model?

### MLP and CNN models. Trainer class is defined after that
class DQNMLP(nn.Module):
    def __init__(self, input_size, num_actions, hidden_layers_sizes):
        """
        Initialize the MLP for Deep Q-Learning.
        1
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



### SIMPLE TRAINER FUNCTION
def train_dqn(self):
    # Initialize loss function and optimizer
    learning_rate, alpha, gamma = self.learning_rate, self.alpha, self.gamma
    replay_buffer = self.experience_buffer
    device = self.device

    # Define the target and online networks
    target_network = self.model.to(device)### theta minus used for calculating the target Q values
    online_network = self.online_model.to(device)### theta in minh et al ### trained every game step

    batch_size = min(len(replay_buffer), self.batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(online_network.parameters(), lr=learning_rate)

    # Sample a mini-batch from the replay buffer
    minibatch = random.sample(replay_buffer, batch_size)

    # Separate the minibatch into states, actions, rewards, and next states
    state_batch, action_batch, next_state_batch, reward_batch = zip(*minibatch)

    # Convert to tensors
    state_batch = torch.stack(state_batch).to(device)
    action_batch = torch.tensor(action_batch, dtype=torch.long).to(device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)

    # Handle next_state_batch for cases where entries could be None
    non_final_mask = torch.tensor([s is not None for s in next_state_batch], dtype=torch.bool, device=device)
    non_final_next_states = torch.stack([s for s in next_state_batch if s is not None]).to(device)

    # Compute Q values for the current state
    q_values = online_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Compute target Q values
    target_q_values = reward_batch.clone()
    with torch.no_grad():
        next_q_values = torch.zeros(batch_size, device=device)
        if len(non_final_next_states) > 0:
            next_q_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]
        print(f"next_q_values.shape: {next_q_values.shape}")

        target_q_values[non_final_mask] += gamma * next_q_values[non_final_mask]

    # Compute loss
    loss = criterion(q_values, target_q_values)
    #print(f"loss is: {loss}") ### I can try to print this loss as well

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_target_network(self):
    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    '''
    self.model = copy.deepcopy(self.online_model)

