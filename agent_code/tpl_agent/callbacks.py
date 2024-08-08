import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .states_to_features import state_to_features
from .neural_agent import DQNCNN, DQNMLP


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    print("First checking Callbacks: setup")
    input_size = 8 * 8 * 6  # Adjust this based on input size (example: 8x8 field with 6 channels)
    num_actions = len(ACTIONS)
    hidden_layers_sizes = [128, 128, 64]  # Example hidden layer sizes

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DQNMLP(input_size, num_actions, hidden_layers_sizes)
    else:
        self.logger.info("Loading model from saved state.")
        self.model = DQNMLP(input_size, num_actions, hidden_layers_sizes)
        self.model.load_state_dict(torch.load("my-saved-model.pt"))
    


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.

    TODO: 
    1. (DONE) check the logic of exploration exploitation. 
    2. Implement linear annealing schedule to the the epsilon. For this I would have to save a number somewhere i feel like. 
    """
    # todo Exploration vs exploitation
    #random_prob = .1
    epsilon = 0.99  # Exploration rate

    if self.train:
        if random.random() < epsilon:
            self.logger.debug("Choosing action purely at random for exploration.")
            return np.random.choice(ACTIONS)
        else:
            self.logger.debug("Choosing action based on model prediction for exploitation.")
            #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0.0])### PLACEHOLDER
            # TODO: I NEED TO CHECK HOW TO INCORPORATE ACTION IN THE NN MODEL FOR DQN. THEN I WILL FILL THIS PART OF THE CODE
            state = state_to_features(game_state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.model(state)
            action_index = torch.argmax(q_values).item()
            return ACTIONS[action_index]
    else:
        # Placeholder for non-training mode
        ### TODO: ADD CODE TO RETURN ACTION USING THE TRAINED Q-NN MODEL
        self.logger.debug("Choosing action using trained model (non-training mode).")
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0.0])
        state = state_to_features(game_state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.model(state)
        action_index = torch.argmax(q_values).item()
        return ACTIONS[action_index]
