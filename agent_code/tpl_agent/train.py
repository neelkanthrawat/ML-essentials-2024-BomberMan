from collections import namedtuple, deque

import pickle
import os
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim

import events as e
# from .callbacks import state_to_features
from .states_to_features import state_to_features, state_to_features_encoder
import copy
from .neural_agent import train_dqn, update_target_network

# For training
TRAIN_FROM_THE_SCRATCH=True ### for subsequent subtasks (2,3,4), we won't start training from the scratch. We will continue training our previously trained model
AGENT_SAVED = 'my-saved-model-17x17-local-state-info.pt'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
experience_buffer = []

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_DICT = {'UP':0,'RIGHT':1,'DOWN':2,'LEFT':3,'WAIT':4,'BOMB':5}

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 500  # keep only ... last transitions 
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events s
VALID_ACTION = "VALID_ACTION"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    print("Then this would be called: train.py - setup_training")
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)# only stores recent transitions
    self.experience_buffer = deque(maxlen=TRANSITION_HISTORY_SIZE) #[] #store all the transitions so far
    #self.device= 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu") # placeholder values

    if not TRAIN_FROM_THE_SCRATCH:# note: self.model is Q_(theta minus) in Mnih et. al (2015) (target network)
        self.model =  self.model.to(self.device)
        self.model.load_state_dict(torch.load(AGENT_SAVED))
    self.online_model = copy.deepcopy(self.model) # denoted as Q_(theta) in Mnih et al. 2015 DQN (online network)
    
    ## NOTE: target network will be updated with online network at the end of the round
    # and online network will be updated every game step in the game_events_occured

    ### defining the network training parameters:
    self.learning_rate =0.001 ### 60k 0.0005 was the learning rate chosen
    self.alpha, self.gamma= 0.9,0.95 #placeholder values, later we would have to optimize over these values
    self.batch_size= 20 #We have also defined the batch size here. Cool!

    ### Initialize epsilon for the epsilon-greedy strategy
    self.epsilon_start = 1.0  # Initial epsilon
    self.epsilon_end = 0.5   # Final epsilon
    self.epsilon_decay = 0.9999995  # Decay factor per step
    self.epsilon = self.epsilon_start  # Start epsilon with the initial value

    ### for the time being we are also using rule based agent to assist us in faster traning
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, 
                        new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards. this is not correct
    if ... :
        pass

    #calculate reward from events
    reward = reward_from_events(self,events)
    # states_to_features is defined in states_to_features.py
    transition_info = Transition(state_to_features_encoder(self,old_game_state), ACTIONS_DICT[self_action], 
                                state_to_features_encoder(self,new_game_state), reward)
    self.transitions.append(transition_info)# Add datum to deque
    self.experience_buffer.append(transition_info)# add datum to list

    ### train the online model
    train_dqn(self)
    ## we can also update the target network here after every C steps. Need to discuss this!

    


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    transition_info = Transition(state_to_features_encoder(self,last_game_state), ACTIONS_DICT[last_action], None, reward)
    # storing the data in deque and list
    self.experience_buffer.append(transition_info)# Add the final transition list
    self.transitions.append(transition_info)# Add the final transition to the deque
    
    ## train the online model 
    train_dqn(self)
    ## update the target network
    update_target_network(self)
    
    # Store the model
    with open(AGENT_SAVED, "wb") as file:
        torch.save(self.model.state_dict(), AGENT_SAVED) ### i need to save the model for future
        #print(f"\nmodel will be saved here after this round")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.

    TODO: 
    1) (DONE) ADD THE FOLLOWING:
    Collect Coin 5
    make a Valid move -1 # all the actions
    make an invalid move: (NEED TO IMPLEMENT IT LOGICALLY -100)
    Kill a player 1000, 700, 500
    Die 1. killed by its own bomb: -1000 and 2. killed by other players: -2000
    Break a crate 2 ### THIS IS IMPORTANT TO FIGURE OUT FOR THE FINAL GAME
    
    2) CHECK: what happens if we remove VALID_ACTION AND INVALID_ACTION:
        REASON: the model should automatically be able to learn taking valid action from other rewards
        such as : collecting coins, breaking crates, killing the opponent.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 400,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -2, 
        e.BOMB_DROPPED: -5,
        e.INVALID_ACTION: -10,
        e.KILLED_SELF: -200,
        e.GOT_KILLED: -1000,
        e.CRATE_DESTROYED: 30
    } 
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

### possible rewards: A* search!
# Collect Coin
# make a Valid move
# make an invalid move: HOW TO IMPLEMENT IT LOGICALLY
# Kill a player
# Die 1. killed by its own bomb and 2. killed by other players
# Break a crate
