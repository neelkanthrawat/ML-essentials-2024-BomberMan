import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# TODO: 
# 1. DQNMLP: change the shape of the output layer from num_actions to 1.
# 2. DQNCNN: figure out how to incorporate action into model?

### MLP and CNN models. Trainer class is defined after that
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

#### defining the trainer class:
class ReplayBufferDataset(Dataset):
    """
    A custom PyTorch Dataset for the replay buffer.
    
    This dataset allows for easy access to experience tuples stored in the replay buffer.
    Each experience tuple contains:
        - state
        - action
        - reward
        - next_state
    
    Attributes:
        replay_buffer (list): The replay buffer containing experience tuples.
    """
    def __init__(self, replay_buffer):
        """
        Initializes the dataset with the replay buffer.
        
        :param replay_buffer: List of experience tuples (state, action, reward, next_state).
        """
        self.replay_buffer = replay_buffer

    def __len__(self):
        """
        Returns the number of experiences in the replay buffer.
        
        :return: The number of experience tuples in the replay buffer.
        """
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        """
        Retrieves an experience tuple at a given index.
        
        :param idx: The index of the experience tuple to retrieve.
        :return: The experience tuple at the given index.
        """
        return self.replay_buffer[idx]

class DQNTrainer:
    """
    A trainer class for Deep Q-Learning with PyTorch.
    
    This class manages the training loop for a DQN model using a replay buffer. It handles:
        - Training over multiple epochs
        - Computing Q-values and loss
        - Updating the model's parameters

    Attributes:
        model (nn.Module): The neural network model for DQN.
        replay_buffer (ReplayBufferDataset): The dataset containing experience tuples.
        learning_rate (float): The learning rate for the optimizer.
        gamma (float): The discount factor for future rewards.
        batch_size (int): The size of each mini-batch.
        epochs (int): The number of epochs to train for.
        steps_per_epoch (int): The number of steps to perform in each epoch.
        print_after (int): How often to print training statistics.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the training on (CPU or GPU).
        avg_train_loss_epoch (list): List to store average training loss for each epoch.
    """
    def __init__(self, model, replay_buffer, learning_rate, gamma, batch_size, epochs, steps_per_epoch, print_after):
        """
        Initializes the trainer with the model, replay buffer, and training parameters.
        
        :param model: The neural network model for DQN.
        :param replay_buffer: The replay buffer dataset.
        :param learning_rate: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        :param batch_size: The size of each mini-batch.
        :param epochs: The number of epochs to train for.
        :param steps_per_epoch: The number of steps to perform in each epoch.
        :param print_after: How often to print training statistics.
        """
        self.model = model
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.print_after = print_after

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.avg_train_loss_epoch = []

    def train(self):
        """
        Main training loop that runs through multiple epochs.
        """
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % self.print_after == 0:
                print("_" * 30)
                print(f"Avg epoch train loss: {self.avg_train_loss_epoch[epoch]:.6f}")
                print("_" * 30)

    def train_epoch(self, epoch):
        """
        Trains the model for a single epoch.
        
        This method processes data in mini-batches, computes Q-values and target Q-values,
        updates model parameters, and logs training progress.
        
        :param epoch: The current epoch number.
        """
        self.model.train()
        total_train_loss = 0
        dataset = ReplayBufferDataset(self.replay_buffer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        tbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_idx, (state_batch, action_batch, reward_batch, next_state_batch) in enumerate(tbar):
            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            reward_batch = reward_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)

            # Compute Q values for the current states
            q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.model(next_state_batch).max(1)[0]
                target_q_values = reward_batch + self.gamma * next_q_values

            # Compute loss
            loss = self.criterion(q_values, target_q_values)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
            tbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(dataloader)
        self.avg_train_loss_epoch.append(avg_train_loss)
