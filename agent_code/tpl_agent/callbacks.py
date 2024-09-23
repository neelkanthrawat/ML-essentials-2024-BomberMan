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
    AGENT_SAVED ='No_loop_anymore_lc_9x9_64x64_2.pt'#'my-saved-model-7x7-local-state-info-rule-based-train-trial.pt'
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
    hidden_layers_sizes = [64,64]#[64,32,32,32,32,32,32,32]#[32,16,8]#[12,8]  # Example hidden layer sizes hidden layer (older):[64,16]
    ### add info for the autoencoder; we will use it for feature reduction and space representation
    # hidden_layers_ae_list = [124*2]### this was when we were using a linear AE
    num_channels_hidden_layer= [16,32]
    code_space_dim_ae = 20# 5x5
    self.device= 'cpu' ### SETTING UP THE DEVICE ### LATER WE WILL CHANGE IT TO CUDA
    self.conv_AE_encoded_features = RETURN_2D_FEAT
    self.with_encoder = WITH_ENCODER
    self.drop_bomb =2
    self.bomb_flag=0
    self.recent_states = deque(maxlen=6)### to avoid looping
    # some varaible we need for aux
    self.close_to_crate, self.prev_close_to_crate=100,100# initialise to very high value
    self.close_to_safe_tile, self.prev_close_to_safe_tile = 0,0### initialise to 0
    if self.train or not os.path.isfile(AGENT_SAVED): # if training phase
        self.logger.info("Setting up model from scratch.")
        if self.with_encoder: ### if you have used encoder for state representation
            self.model = DQNMLP(code_space_dim_ae, num_actions, hidden_layers_sizes)
            ### instantiate the AE
            self.ae =ConvAutoencoder(input_channels=6,bottleneck_dim=code_space_dim_ae,
                                    image_size=image_size, num_channels_hidden_layer=num_channels_hidden_layer)
            self.ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=torch.device(self.device)))
            self.ae.eval()
            ### we did not include the case where we use MLP based AE, because for the time being, we dont use it
        else: ### for the 1-D naive / hand crafted reduced feature vector
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
        else:### 1-D naive states/ hand crafted features
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
    2. Implement linear annealing schedule to the the epsilon. For this I would have to save a number somewhere i feel like. 
    """
    # todo Exploration vs exploitation 
    #random_prob = .1
    # If training, use the diminishing epsilon-greedy strategy
    if self.train:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        prob = random.random()
        if prob < self.epsilon: ### 
            
            # actions_for_coin_collection=['UP', 'RIGHT', 'DOWN', 'LEFT']
            # action=np.random.choice(ACTIONS)
            # return action
            # if self.drop_bomb%5 == 0:#np.random.choice([7,11,15])
            #     self.drop_bomb+=1
            #     return 'BOMB'
            # else:
            # self.drop_bomb+=1
            if 0.0<=prob<0.95*self.epsilon:#prob < 0.*self.epsilon:
                # print("random")
                self.logger.debug("Choosing action purely at random for exploration.")
                return np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB'], p=[0.2,0.2,0.2,0.2,0.2])
            elif 0.95*self.epsilon<=prob < self.epsilon: #0.0*self.epsilon<=prob<self.epsilon:
                # print("rule based")
                action = act_rule_based(self, game_state=game_state)
                if action ==None: #or action == 'WAIT':
                    action = np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT'])
                return action
            
        else:
            # print("agent")
            self.logger.debug("Choosing action based on model prediction for exploitation.")
            # state = state.clone().detach().unsqueeze(0).float()#torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            state = state_to_features_encoder(self, game_state, check_close_to_crate=False)
            with torch.no_grad():
                q_values = self.model(state)
            # print(f"q values.shape: {q_values}")
            
            if not PROB_TRAIN:
                ### deterministic strategy
                action_index = torch.argmax(q_values).item()    # If no free tile is found, re=self, game_state=game_state)
                # print(f"action is: {ACTIONS[action_index]}")
                return ACTIONS[action_index]
            elif PROB_TRAIN:
                min_q = torch.min(q_values)
                max_q = torch.max(q_values)
                c = torch.dist(min_q, max_q)/2#torch.abs(torch.mean(q_values))### extra step added
                
                norm_q = (q_values - min_q) / (max_q - min_q) *c # PROB_ACTION_SCALAR
                # print(f"prob()norm_q_* c): {torch.softmax(norm_q*c, dim=0)}")
                prob = torch.softmax(norm_q, dim=0)
                
            
                action_index = torch.multinomial(prob, 1)
                
                return ACTIONS[action_index]
    else:
        # Placeholder for non-training mode
        ### TODO: ADD CODE TO RETURN ACTION USING THE TRAINED Q-NN MODEL
        # print(f"STEP NUMBER: {self.count}")
        # self.count+=1
        self.model.eval()
        self.logger.debug("Choosing action using trained model (non-training mode).")
        # if self.drop_bomb%10 == 0:
        #         self.drop_bomb+=1
        #         return 'BOMB'
        # else:
            # self.drop_bomb+=1
        ### filling in the state  in recent state
        # print("recent state is:",game_state['self'][3])
        self.recent_states.append(game_state['self'][3])
        state = state_to_features_encoder(self, game_state)
        with torch.no_grad():
            q_values = self.model(state)
        # print(f"q values are: {q_values}")
        # print(f"actions list: {ACTIONS}")
        if not PROB_ACTION_ON:
            ## deterministic policy
            action_index = torch.argmax(q_values).item()
            self_action = ACTIONS[action_index]
            # print(f"action is: {self_action}")
            return self_action
        elif PROB_ACTION_ON:
        ### Let's work with probabilistic policy
            # min_q = torch.min(q_values)
            # max_q = torch.max(q_values)
            # norm_q = (q_values - min_q)/(max_q - min_q)*7#PROB_ACTION_SCALAR
            # prob = torch.softmax(norm_q,dim=0)
            # print(f"actions list: {ACTIONS}")
            # print(f"q values (stochastic) are: {q_values}")
            # print(f"prob vector is: {prob}")
            # action_index = torch.multinomial(prob,1)
            # print(f"action it took: {ACTIONS[action_index]}")
            # return ACTIONS[action_index] 
            # Normalize the q_values
            min_q = torch.min(q_values)
            max_q = torch.max(q_values)
            # c = torch.dist(min_q, max_q)/2#torch.abs(torch.mean(q_values))### extra step added
            print(f"q values: {q_values}")
            norm_q = (q_values - min_q) / (max_q - min_q)# * 7 # PROB_ACTION_SCALAR
            print(f"normalised q: {norm_q}")
            # print(f"prob()norm_q_* c): {torch.softmax(norm_q*c, dim=0)}")
            prob = torch.softmax(norm_q, dim=0)
            if TOP_K:
                # Zero out all but the top 2 probabilities
                topk_indices = torch.topk(prob, 2).indices
                mask = torch.zeros_like(prob)
                mask[topk_indices] = 1
                masked_prob = prob * mask
                masked_prob /= masked_prob.sum()  # Normalize to ensure it sums to 1
                action_index = torch.multinomial(masked_prob, 1)
                # return ACTIONS[action_index]
            else:
                action_index = torch.multinomial(prob, 1)
            return ACTIONS[action_index]