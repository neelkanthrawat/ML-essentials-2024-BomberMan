from collections import namedtuple, deque

import numpy as np
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
TRAIN_FROM_THE_SCRATCH= 1### for subsequent subtasks (2,3,4), we won't start training from the scratch. We will continue training our previously trained model
AGENT_SAVED =   'No_loop_anymore_lc_9x9_64x64_2.pt'#'my-saved-model-7x7-local-state-info-rule-based-train-trial.pt'#'my-saved-model-7x7-local-state-info-3-trial.pt' #'my-saved-model-17x17-local-state-a-star-info.pt'#'my-saved-model-17x17-local-state-info.pt'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
experience_buffer = []

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_DICT = {'UP':0,'RIGHT':1,'DOWN':2,'LEFT':3,'WAIT':4,'BOMB':5}

#### define custom events
CLOSE_T0_CRATE = 'CLOSE TO CRATE'
AWAY_FROM_CRATE = 'AWAY FROM CRATE'
CLOSE_TO_SAFE_TILE = 'CLOSE TO SAFE TILE'
AWAY_FROM_SAFE_TILE = 'AWAY FROM SAFE TILE'
BOMB_NEAR_CRATE = 'BOMB NEAR CRATE'
BOMB_FAR_FROM_CRATE = 'BOMB_FAR_FROM_CRATE'
BOMB_DROPPED_NAIVELY = 'BOMB_DROPPED_NAIVELY'
AGENT_LOOP = 'AGENT_LOOP'
STUPID_BOMB= 'STUPID_BOMBING'


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000#3000  # keep only ... last transitions 
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
    #print("Then this would be called: train.py - setup_training")
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)# only stores recent transitions
    self.experience_buffer = deque(maxlen=TRANSITION_HISTORY_SIZE) #[] #store all the transitions so far
    self.device= 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu") # placeholder values
    self.n_step=3# idk what this is for tbh
    if not TRAIN_FROM_THE_SCRATCH:# note: self.model is Q_(theta minus) in Mnih et. al (2015) (target network)
        self.model =  self.model.to(self.device)
        self.model.load_state_dict(torch.load(AGENT_SAVED))
    self.online_model = copy.deepcopy(self.model) # denoted as Q_(theta) in Mnih et al. 2015 DQN (online network)
    
    ## NOTE: target network will be updated with online network at the end of the round
    # and online network will be updated every game step in the game_events_occured

    ### defining the network training parameters:
    self.learning_rate =0.001 ### 60k 0.0005 was the learning rate chosen
    self.alpha, self.gamma= 0.9,0.65#0.65#0.95 #placeholder values, later we would have to optimize over these values
    self.batch_size= 500#64#20(it was 20 even for local state representation) #We have also defined the batch size here. Cool!
    #### i changed gamma to 0.5 and it stopped working. only o/p was waiting
    ### Initialize epsilon for the epsilon-greedy strategy
    self.epsilon_start =1#1 # Initial epsilon
    self.epsilon_end = 0.1#0.05#0.1   # Final epsilon
    self.epsilon_decay = 0.9999985#0.9999985#(13x13)#0.999985# (9x9)#0.9999985#0.999999885#0.9999995  # Decay factor per step
    self.epsilon = self.epsilon_start  # Start epsilon with the i,
    self.current_round = 0
    
    ### some variables we need for auxiliary reward:
    self.close_to_crate, self.prev_close_to_crate=100,100# initialise to very high value
    self.close_to_safe_tile, self.prev_close_to_safe_tile = 0,0### initialise to 0
    self.prev_action = None
    self.old_state = None
    self.step_count=0
    self.recent_states = deque(maxlen=6)# earlier it was 20
    self.recent_actions = deque(maxlen=6)
    self.no_crate_nearby=0
    self.no_coin_nearby=0
    ### some flags
    self.bomb_flag=0
    self.close_to_crate_flag=0
    self.far_from_crate_flag=0
    self.prev_bomb_counter=6

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
    # print("_"*10)
    # print("___before adding custom events___")
    # print(f"self_action: {self_action}")
    # #print(f"old position: {(old_game_state['self'][3][1],old_game_state['self'][3][0]) }")
    # print("before event reviewing")
    # print(f"self.close_to_crate: {self.close_to_crate}")
    # print(f"self.prev_close_to_crate: {self.prev_close_to_crate}")
    # print(f"self.close_to_safe_tile: {self.close_to_safe_tile}")
    # print(f"self.prev_close_to_safe_tile: {self.prev_close_to_safe_tile}")
    
    ### add custom events here

    ## to ensure bombing at appropriate positions
    # print("testing game events")
    if self.prev_close_to_crate==100 and self.close_to_crate==0 and self_action=='BOMB':
        events.append(BOMB_DROPPED_NAIVELY)
        # print("DROPPING BOMB NAIVELY")
        self.bomb_flag = 1
    elif self.prev_close_to_crate==1 and self.close_to_crate == 0  and self_action =='BOMB':
        events.append(BOMB_NEAR_CRATE)
        # print("DROPPING BOMB NAIVELY")
        # print("bomb near crate")
        self.bomb_flag=1
    elif self.close_to_crate!=0 and self_action =='BOMB':
        events.append(BOMB_FAR_FROM_CRATE)
        # print("bomb far from crate")
        self.bomb_flag = 1
    
    if self.bomb_flag:
        self.close_to_crate=100
        self.prev_close_to_crate=100
    
    ### if no crate nearby but still bombing instead of navigating further
    # if self.no_crate_nearby and self_action == 'BOMB':
    #     events.append(STUPID_BOMB)
    #     self.no_crate_nearby=0

    ## to endure agent comes close to the crate
    if self.close_to_crate < self.prev_close_to_crate:# reward if agent moves closer to the crate
        if self.bomb_flag ==0:
            if self.close_to_crate==0 and self.prev_close_to_crate ==100:
                ### takes care of the case when no crate in the sight OR after the bomb is dropped, before it explodes
                # print("takes care of the case when no crate in the sight OR   after the bombing but before explosion")
                self.close_to_crate = 100
            else:
                # events.append(CLOSE_T0_CRATE)
                # print("we got close to crate")
                self.prev_close_to_crate = self.close_to_crate
    elif self.close_to_crate > self.prev_close_to_crate:# penalise if agent moves away from closest crate
        # events.append(AWAY_FROM_CRATE)
        # print("went far from crate")
        self.prev_close_to_crate = self.close_to_crate
    self.bomb_flag =0
    # print(f"AFTER")
    # print(f"self.close_to_crate: {self.close_to_crate}")
    # print(f"self.prev_close_to_crate: {self.prev_close_to_crate}")
    
    
    (x, y) = new_game_state['self'][3]
    # if self.recent_states.count((x,y)) > 2:
    #     events.append(AGENT_LOOP)
    #     # print(f"loops are present as self.recent_states.count((x,y)) is {self.recent_states.count((x,y))}")
    #     # print(f"for current round number: {self.current_round} and step: {self.step_count}")
    #     # print(f" it loops")
    self.recent_states.append((x,y))
    
    
        
    #calculate reward from events
    reward = reward_from_events(self,events)
    self.prev_action=self_action
    # states_to_features is defined in states_to_features.py
    state_old_game=state_to_features_encoder(self,old_game_state)
    state_new_game = state_to_features_encoder(self,new_game_state)
    ###
    transition_info = Transition(state_old_game, ACTIONS_DICT[self_action], state_new_game
                                , reward)
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
    if self.current_round%10:
        update_target_network(self)
    
        # Store the model
        with open(AGENT_SAVED, "wb") as file:
            torch.save(self.model.state_dict(), AGENT_SAVED) ### i need to save the model for future
            ##print(f"\nmodel will be saved here after this round")
        
    ### reset the values for the new round
    self.close_to_crate, self.prev_close_to_crate=100,100# initialise to very high value
    self.close_to_safe_tile, self.prev_close_to_safe_tile = 0,0### initialise to 0
    self.step_count=0
    self.current_round+=1
    self.recent_states.clear()
    # print("self.recent_states:", self.recent_states)

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
        e.COIN_COLLECTED: 10000,# earlier it was 100
        e.KILLED_OPPONENT: 400,
        e.MOVED_RIGHT:200,
        e.MOVED_LEFT: 200,
        e.MOVED_UP: 200,
        e.MOVED_DOWN: 200, 
        
        e.WAITED: -300,#-300, ### up until now I was using -2
        e.BOMB_DROPPED:2900,#2900,#1#-1,# -5 # 50# initally it was 3500
        e.INVALID_ACTION: -4000,#-100,# earlier it was -10 
        e.KILLED_SELF: -5000,# earlier it was -800
        e.GOT_KILLED: -3000,
        e.CRATE_DESTROYED: 800,# earlier it was 30
        ### we need to add some more custom rewards here. else it won't work
        CLOSE_T0_CRATE : 150,
        CLOSE_TO_SAFE_TILE : 150,# not using this
        BOMB_NEAR_CRATE : 4000,#3500,
        BOMB_FAR_FROM_CRATE: -3200,#-1000,#-1000,
        BOMB_DROPPED_NAIVELY: -4200,#-3000,
        AWAY_FROM_CRATE: -3050,# it was -20
        AWAY_FROM_SAFE_TILE: -2000,### not using this as well
        STUPID_BOMB: -800,
        AGENT_LOOP: -5000# NOT USING IT
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


'''e.COIN_COLLECTED: 200,# earlier it was 100
        e.KILLED_OPPONENT: 400,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -50, ### up until now I was using -2
        e.BOMB_DROPPED: 50,#-1,# -5 # 50
        e.INVALID_ACTION: -1000,#-100,# earlier it was -10 
        e.KILLED_SELF: -500,
        e.GOT_KILLED: -1000,
        e.CRATE_DESTROYED: 100,# earlier it was 30
        ### we need to add some more custom rewards here. else it won't work
        CLOSE_T0_CRATE : 10,
        CLOSE_TO_SAFE_TILE : 10,
        BOMB_NEAR_CRATE : 50,
        AWAY_FROM_CRATE: -15,
        AWAY_FROM_SAFE_TILE: -15'''
        
#### this reward seems to be working, although it sometimes get's stuck in a loop
'''
game_rewards = {
        e.COIN_COLLECTED: 200,# earlier it was 100
        e.KILLED_OPPONENT: 400,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -20, ### up until now I was using -2
        e.BOMB_DROPPED:-20,#1#-1,# -5 # 50
        e.INVALID_ACTION: -1000,#-100,# earlier it was -10 
        e.KILLED_SELF: -800,# earlier it was -800
        e.GOT_KILLED: -1000,
        e.CRATE_DESTROYED: 100,# earlier it was 30
        ### we need to add some more custom rewards here. else it won't work
        CLOSE_T0_CRATE : 15,
        CLOSE_TO_SAFE_TILE : 15,
        BOMB_NEAR_CRATE : 75,
        BOMB_FAR_FROM_CRATE: -80,
        AWAY_FROM_CRATE: -50,# it was -20
        AWAY_FROM_SAFE_TILE: -50 ### udate in the train: it worked well with bomb=1 again!
        
    } '''