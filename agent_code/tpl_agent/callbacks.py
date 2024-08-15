import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .states_to_features import state_to_features, state_to_features_encoder
from .neural_agent import DQNCNN, DQNMLP
from .rule_based_agent_action import rule_based_action, look_for_targets, reset_self
from .autoencoder_feature_reduction import Autoencoder


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
AE_WEIGHTS_PATH = 'D:\\Desktop\\master_scientific_computing\\second_semester\\ml essentials\\Final Project\\ML-essentials-2024-BomberMan\\ae_model_weights_7x7.pth'
AGENT_SAVED = 'my-saved-model-7x7-with-ae.pt'

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
    input_size = 7 * 7 * 6  # Adjust this based on input size (example: 8x8 field with 6 channels)
    num_actions = len(ACTIONS)
    hidden_layers_sizes = [12,8]  # Example hidden layer sizes hidden layer (older):[64,16]
    ### add info for the autoencoder; we will use it for feature reduction and space representation
    hidden_layers_ae_list = [124*2]
    code_space_dim_ae = 20
    self.device= 'cpu' ### SETTING UP THE DEVICE ### LATER WE WILL CHANGE IT TO CUDA
    if self.train or not os.path.isfile(AGENT_SAVED):
        self.ae = Autoencoder(input_size,hidden_layers_ae_list, code_space_dim_ae)
        self.logger.info("Setting up model from scratch.")
        self.model = DQNMLP(code_space_dim_ae, num_actions, hidden_layers_sizes)
        self.ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=torch.device(self.device)))
        self.ae.eval()
    else:
        self.logger.info("Loading model from saved state.")
        self.model = DQNMLP(code_space_dim_ae, num_actions, hidden_layers_sizes)
        self.model.load_state_dict(torch.load(AGENT_SAVED))
        self.model.eval()
        self.ae = Autoencoder(input_size,hidden_layers_ae_list, code_space_dim_ae)
        self.ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=torch.device(self.device)))
        self.ae.eval()
    


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
    # If training, use the diminishing epsilon-greedy strategy
    if self.train:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        if random.random() < self.epsilon: ### 
            self.logger.debug("Choosing action purely at random for exploration.")
            actions_for_coin_collection=['UP', 'RIGHT', 'DOWN', 'LEFT']
            action=np.random.choice(actions_for_coin_collection)
            # print(f"\nACTION: {action}")
            return action
        else:
            # using rule based agent to assist us in faster training
            # self.logger.debug("Choosing action based on rule-based logic for exploitation.")
            # return rule_based_action(self,game_state)
            self.logger.debug("Choosing action based on model prediction for exploitation.")
            #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0.0])### PLACEHOLDER
            # TODO: I NEED TO CHECK HOW TO INCORPORATE ACTION IN THE NN MODEL FOR DQN. THEN I WILL FILL THIS PART OF THE CODE
            # state = state_to_features(game_state)
            # state = state.clone().detach().unsqueeze(0).float()#torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            state = state_to_features_encoder(self, game_state)
            with torch.no_grad():
                q_values = self.model(state)
            action_index = torch.argmax(q_values).item()
            # print(f"\nAction is {ACTIONS[action_index]}")
            return ACTIONS[action_index]
    else:
        # Placeholder for non-training mode
        ### TODO: ADD CODE TO RETURN ACTION USING THE TRAINED Q-NN MODEL
        self.model.eval()
        self.logger.debug("Choosing action using trained model (non-training mode).")
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, 0.0])
        # state = state_to_features(game_state)
        # state = state.clone().detach().unsqueeze(0).float()#torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        state = state_to_features_encoder(self, game_state)
        with torch.no_grad():
            q_values = self.model(state)
        action_index = torch.argmax(q_values).item()
        # print(f"\nAction is {ACTIONS[action_index]}")
        return ACTIONS[action_index]
