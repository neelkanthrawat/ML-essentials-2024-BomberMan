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
TRAIN_FROM_THE_SCRATCH= 1 ### for subsequent subtasks (2,3,4), we won't start training from the scratch. We will continue training our previously trained model
AGENT_SAVED = 'complex44_2.pt'#'my-saved-model-7x7-local-state-info-rule-based-train-trial.pt'#'my-saved-model-7x7-local-state-info-3-trial.pt' #'my-saved-model-17x17-local-state-a-star-info.pt'#'my-saved-model-17x17-local-state-info.pt'
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
AGENT_LOOP = 'AGENT_LOOP'


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
    #print("Then this would be called: train.py - setup_training")
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
    self.alpha, self.gamma= 0.9,0.6#0.65#0.95 #placeholder values, later we would have to optimize over these values
    self.batch_size= 64#20(it was 20 even for local state representation) #We have also defined the batch size here. Cool!
    #### i changed gamma to 0.5 and it stopped working. only o/p was waiting
    ### Initialize epsilon for the epsilon-greedy strategy
    self.epsilon_start = 1.0  # Initial epsilon
    self.epsilon_end = 0.8#0.1   # Final epsilon
    self.epsilon_decay = 0.99999995#0.9999995  # Decay factor per step
    self.epsilon = self.epsilon_start  # Start epsilon with the i,
    self.current_round = 0
    
    ### some variables we need for auxiliary reward:
    self.close_to_crate, self.prev_close_to_crate=100,100# initialise to very high value
    self.close_to_safe_tile, self.prev_close_to_safe_tile = 0,0### initialise to 0
    self.prev_action = None
    self.old_state = None
    self.step_count=0

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
    #print("___before adding custom events___")
    #print(f"old position: {(old_game_state['self'][3][1],old_game_state['self'][3][0]) }")
   
    #print(f"self.close_to_crate: {self.close_to_crate}")
    #print(f"self.prev_close_to_crate: {self.prev_close_to_crate}")
    #print(f"self.close_to_safe_tile: {self.close_to_safe_tile}")
    #print(f"self.prev_close_to_safe_tile: {self.prev_close_to_safe_tile}")
    
    ### adding custom events here
    if self.close_to_crate < self.prev_close_to_crate:# reward if agent moves closer to the crate
        if self.close_to_crate==0 and self.prev_close_to_crate ==100:
            ### takes care of the case when no crate in the sight OR after the bombing, before the bomb eplodes
            # #print("takes care of the case when no crate in the sight OR   after the bombing but before explosion")
            self.close_to_crate = 100
        else:
            events.append(CLOSE_T0_CRATE)
            # #print("we got close to crate")
            self.prev_close_to_crate = self.close_to_crate
    elif self.close_to_crate > self.prev_close_to_crate:# penalise if agent moves away from closest crate
        events.append(AWAY_FROM_CRATE)
        #print("went far from crate")
        self.prev_close_to_crate = self.close_to_crate
    
    if self.prev_close_to_crate == 0 and self_action == 'BOMB':# big reward if bomb is dropped close to the crate
       ### give a bog reward if the bomb is places near the crate
        #print("dropped bomb near the crate")
        events.append(BOMB_NEAR_CRATE)
        # reset
        self.prev_close_to_crate=100### i think we should reset this to 0
        self.close_to_crate= 100
    elif self.prev_close_to_crate!=0 and self_action == 'BOMB':###else penalise if bomb dropped far away
        #print("dropped bomb far away from the crate")
        events.append(BOMB_FAR_FROM_CRATE)
        # reset again
        self.prev_close_to_crate=100# I think this should be reset to 0
        self.close_to_crate=100
        
    # if self.close_to_safe_tile < self.prev_close_to_safe_tile:# reward if agent moves close to  safe tile
    #     #print("getting close to the safe tile")
    #     events.append(CLOSE_TO_SAFE_TILE)
    #     self.prev_close_to_safe_tile = self.close_to_safe_tile
    # elif self.close_to_safe_tile > self.prev_close_to_safe_tile:### penalise if agent moves away from safe tile
    #     if self.prev_close_to_safe_tile==0 and self.close_to_safe_tile > 0 and self.prev_action=='BOMB':
    #         #THIS TAKES CARE OF SITUATION JUST AFTER THE BOMB
    #         #print("still would be considered close to safe tile as previous action was bombings")
    #         # events.append(CLOSE_TO_SAFE_TILE)
    #         pass
    #     else:
    #         events.append(AWAY_FROM_SAFE_TILE)
    #         #print("away from the safe tile")
    #     self.prev_close_to_safe_tile = self.close_to_safe_tile
    # elif self.close_to_safe_tile == 0:#### this is causing issue.
    #     self.prev_close_to_crate, self.close_to_crate=100, 100
        
    #print("___After  adding custom events___")
    #print(f"self.close_to_crate: {self.close_to_crate}")
    #print(f"self.prev_close_to_crate: {self.prev_close_to_crate}")
    #print(f"self.close_to_safe_tile: {self.close_to_safe_tile}")
    #print(f"self.prev_close_to_safe_tile: {self.prev_close_to_safe_tile}")
    #print(f"action chosen and the new state then would be:")
    #print(f"action is: {self_action}")
    #print(f"current position: {(new_game_state['self'][3][1],new_game_state['self'][3][0]) }")
    #print("_*_*_*_*_*_*")
        

    #calculate reward from events
    reward = reward_from_events(self,events)
    self.prev_action=self_action
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
    
    ### adding custom events here
    
    # if self.close_to_crate < self.prev_close_to_crate:# reward if agent moves closer to the crate
    #     if self.close_to_crate == 0 and self.prev_close_to_crate ==100:
    #         ### takes care of the case when no crate in the sight OR after the bombing, before the bomb explodes
           
    #         self.close_to_crate = 100
    #     else:
    #         events.append(CLOSE_T0_CRATE)
    #         self.prev_close_to_crate = self.close_to_crate
    # elif self.close_to_crate > self.prev_close_to_crate:# penalise if agent moves away from closest crate
    #     events.append(AWAY_FROM_CRATE)
    #     self.prev_close_to_crate = self.close_to_crate
    
    # if self.prev_close_to_crate == 0 and last_action == 'BOMB':# big reward if bomb is dropped close to the crate
    #    ### give a bog reward if the bomb is places near the crate
    #     events.append(BOMB_NEAR_CRATE)
    #     # reset
    #     self.prev_close_to_crate=100### i think we should reset this to 0
    #     self.close_to_crate= 100
    # elif self.prev_close_to_crate!=0 and last_action == 'BOMB':###else penalise if bomb dropped far away
    #     events.append(BOMB_FAR_FROM_CRATE)
    #     # reset again
    #     self.prev_close_to_crate=100# I think this should be reset to 0
    #     self.close_to_crate=100
        
    # if self.close_to_safe_tile < self.prev_close_to_safe_tile:# reward if agent moves close to  safe tile
    #     events.append(CLOSE_TO_SAFE_TILE)
    #     self.prev_close_to_safe_tile = self.close_to_safe_tile
    # elif self.close_to_safe_tile > self.prev_close_to_safe_tile:### penalise if agent moves away from safe tile
    #     if self.prev_close_to_safe_tile==0 and self.close_to_safe_tile > 0 and self.prev_action=='BOMB':
    #         #THIS TAKES CARE OF SITUATION JUST AFTER THE BOMB
    #         # events.append(CLOSE_TO_SAFE_TILE)
    #         pass
    #     else:
    #         events.append(AWAY_FROM_SAFE_TILE)
    #     self.prev_close_to_safe_tile = self.close_to_safe_tile
        
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
        ##print(f"\nmodel will be saved here after this round")
    
    ### reset the values for the new round
    self.close_to_crate, self.prev_close_to_crate=100,100# initialise to very high value
    self.close_to_safe_tile, self.prev_close_to_safe_tile = 0,0### initialise to 0
    self.step_count=0

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
        e.COIN_COLLECTED: 1000,# earlier it was 100
        e.KILLED_OPPONENT: 400,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -200, ### up until now I was using -2
        e.BOMB_DROPPED:500,#1#-1,# -5 # 50
        e.INVALID_ACTION: -1000,#-100,# earlier it was -10 
        e.KILLED_SELF: -800,# earlier it was -800
        e.GOT_KILLED: -1000,
        e.CRATE_DESTROYED: 100,# earlier it was 30
        ### we need to add some more custom rewards here. else it won't work
        CLOSE_T0_CRATE : 15,
        CLOSE_TO_SAFE_TILE : 15,
        BOMB_NEAR_CRATE : 500,
        BOMB_FAR_FROM_CRATE: -800,
        AWAY_FROM_CRATE: -15,# it was -20
        AWAY_FROM_SAFE_TILE: -20,
        AGENT_LOOP: -5
        
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