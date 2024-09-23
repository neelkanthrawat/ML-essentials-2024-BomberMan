import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

### need deque to avoid loop
from collections import namedtuple, deque

from .states_to_features import state_to_features, state_to_features_encoder
from .neural_agent import DQNCNN, DQNMLP
from .rule_based_agent_action import act_rule_based
from .autoencoder_feature_reduction import Autoencoder, ConvAutoencoder

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
AE_WEIGHTS_PATH = os.path.join(current_dir,'files_ae', 'conv2_ae_model_weights_5x5.pth')

WITH_CONV_AE=False
image_size=5 ### local image info size
PROB_ACTION_ON=0#0
PROB_TRAIN=False
TOP_K=True

if WITH_CONV_AE:### shape of the reduced features: torch.Size([1, 20]):
    AGENT_SAVED = 'my-saved-model-5x5-with-conv2-ae.pt'
    RETURN_2D_FEAT=True 
    WITH_ENCODER=True
else: ### shape of the naive features: torch.Size([150])
    AGENT_SAVED ='No_loop_anymore_lc_9x9_64x64_better_escape_bomb.pt'#'my-saved-model-7x7-local-state-info-rule-based-train-trial.pt'
    RETURN_2D_FEAT=False
    WITH_ENCODER=False 

COUNT=1


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
    print(f"----IMAGE SIZE IS: -----: {image_size}")
    input_size = image_size*image_size* 5  #6: when we were testing if the model escapes its own bomb  # Adjust this based on input size (example: 8x8 field with 6 channels)
    num_actions = len(ACTIONS)
    hidden_layers_sizes = [64,64] # THIS IS WHAT WE WILL USE FOR FINAL TOURNAMENT
    ### add info for the autoencoder; we will use it for feature reduction and space representation
    # hidden_layers_ae_list = [124*2]### this was when we were using a linear AE
    num_channels_hidden_layer= [16,32]
    code_space_dim_ae = 20# 5x5
    self.device= 'cpu'
    self.conv_AE_encoded_features = RETURN_2D_FEAT
    self.with_encoder = WITH_ENCODER
    self.drop_bomb =2
    self.bomb_flag=0
    self.recent_states = deque(maxlen=6)### will be used later to avoid looping
    # some varaible we need for aux
    # i think we can delete following 2 lines
    self.close_to_crate, self.prev_close_to_crate=100,100# initialise to very high value
    self.close_to_safe_tile, self.prev_close_to_safe_tile = 0,0### initialise to 0
    if self.train or not os.path.isfile(AGENT_SAVED): # if training phase
        self.logger.info("Setting up model from scratch.")
        if self.with_encoder: ### if you have used encoder for state representation
            # DOES NOT HELP AT ALL. WE WONT USE IT 
            self.model = DQNMLP(code_space_dim_ae, num_actions, hidden_layers_sizes)
            ### instantiate the AE
            self.ae =ConvAutoencoder(input_channels=6,bottleneck_dim=code_space_dim_ae,
                                    image_size=image_size, num_channels_hidden_layer=num_channels_hidden_layer)
            self.ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=torch.device(self.device)))
            self.ae.eval()
            ### we did not include the case where we use MLP based AE, because for the time being, we dont use it
        else: ### for the 1-D  hand crafted reduced feature vector (FOR FINAL TOURNAMENT)
            self.model = DQNMLP(input_size, num_actions, hidden_layers_sizes)
    else: ### if not the training phase
        self.logger.info("Loading model from saved state.")
        if self.with_encoder:### with conv encoder case (self.conv_AE_encoded_features)
            self.model = DQNMLP(code_space_dim_ae, num_actions, hidden_layers_sizes)
            self.ae = ConvAutoencoder(input_channels=6,bottleneck_dim=code_space_dim_ae,
                                image_size=image_size, 
                                num_channels_hidden_layer=num_channels_hidden_layer)
            self.ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=torch.device(self.device)))
            self.ae.eval()
            ### NOTE: Note again that we did not include the case where we use MLP based AE
        else:### 1-D  hand crafted features: WILL BE USED FOR FINAL TOURNAMENT
            self.model = DQNMLP(input_size, num_actions, hidden_layers_sizes)
        self.model.load_state_dict(torch.load(AGENT_SAVED))
        self.model.eval()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.

    TODO: 
    1. (DONE) check the logic of exploration exploitation. 
    2. Implement  annealing schedule to the the epsilon. For this I would have to save a number somewhere i feel like. 
    """
    # todo Exploration vs exploitation 
    # If training, use the diminishing epsilon-greedy strategy
    if self.train:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        prob = random.random()
        if prob < self.epsilon: #exploration step
            if 0.0<=prob<0.95*self.epsilon:#
                # random
                self.logger.debug("Choosing action purely at random for exploration.")
                return np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB'], p=[0.2,0.2,0.2,0.2,0.2])
            elif 0.95*self.epsilon<=prob < self.epsilon: #to see good moves during training
                # "rule based"
                action = act_rule_based(self, game_state=game_state)
                if action ==None:
                    action = np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT'])
                return action
        else:# exploitation step
            self.logger.debug("Choosing action based on model prediction for exploitation.")
            state = state_to_features_encoder(self, game_state, check_close_to_crate=False)
            with torch.no_grad():
                q_values = self.model(state)            
            if not PROB_TRAIN:# this is what we will use for final
                ### deterministic strategy
                action_index = torch.argmax(q_values).item() # If no free tile is found, re=self, game_state=game_state)
                return ACTIONS[action_index]
            elif PROB_TRAIN:
                # probabilistic (turns out to be not so good)
                min_q = torch.min(q_values)
                max_q = torch.max(q_values)
                c = torch.dist(min_q, max_q)/2#torch.abs(torch.mean(q_values))### extra step added
                norm_q = (q_values - min_q) / (max_q - min_q) *c # PROB_ACTION_SCALAR
                prob = torch.softmax(norm_q, dim=0)
                action_index = torch.multinomial(prob, 1)
                return ACTIONS[action_index]
    else:
        # Placeholder for non-training mode
        ### TODO: ADD CODE TO RETURN ACTION USING THE TRAINED Q-NN MODEL
        self.model.eval()
        self.logger.debug("Choosing action using trained model (non-training mode).")
        self.recent_states.append(game_state['self'][3])
        state = state_to_features_encoder(self, game_state)
        with torch.no_grad():
            q_values = self.model(state)
        if not PROB_ACTION_ON:# THIS WILL BE USED FOR FINAL TOURNAMENT
            ## deterministic policy
            action_index = torch.argmax(q_values).item()
            self_action = ACTIONS[action_index]
            return self_action
        elif PROB_ACTION_ON:
        ### probabilistic policy
            # Normalize the q_values
            min_q = torch.min(q_values)
            max_q = torch.max(q_values)
            norm_q = (q_values - min_q) / (max_q - min_q)# * 7 # PROB_ACTION_SCALAR
            prob = torch.softmax(norm_q, dim=0)
            if TOP_K:
                # Zero out all but the top 2 probabilities
                topk_indices = torch.topk(prob, 2).indices
                mask = torch.zeros_like(prob)
                mask[topk_indices] = 1
                masked_prob = prob * mask
                masked_prob /= masked_prob.sum()  # Normalize to ensure it sums to 1
                action_index = torch.multinomial(masked_prob, 1)
            else:
                action_index = torch.multinomial(prob, 1)
            return ACTIONS[action_index]