import numpy as np
import torch
import heapq
from collections import deque
import torch.nn as nn
### this code implements the following:
# 1. naive full game information
# 2. reduced state representation using AE (encoder to be precise)

BLOCK_SIZE=5
CENTER_POS=(2,2)

###
AE_PRETRAINED_FILE='D:\\Desktop\\master_scientific_computing\\second_semester\\ml essentials\\Final Project\\ML-essentials-2024-BomberMan\\ae_model_weights_5x5.pth'

def get_subblock_with_padding(tensor, r, c, block_size=3, padding_value_list=None):
    """
    Extract a sub-block from the input 3D tensor such that the given coordinates (r, c)
    are at the center of the sub-block. If the sub-block goes out of bounds, pad with specific values
    provided in padding_value_list.

    Parameters:
    tensor (torch.Tensor): The large input 3D tensor with shape (num_channels, height, width).
    r (int): Row index of the center.
    c (int): Column index of the center.
    block_size (int): Size of the sub-block (block_size x block_size). Must be odd.
    padding_value_list (list): List of padding values for each channel. Length should match num_channels.

    Returns:
    torch.Tensor: The sub-block centered at (r, c) with shape (num_channels, block_size, block_size),
                  padded with the corresponding values from padding_value_list if necessary.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be an odd number to have a center.")
    
    num_channels, height, width = tensor.shape

    if padding_value_list is None:
        padding_value_list = [5000] * num_channels
    elif len(padding_value_list) != num_channels:
        raise ValueError("Length of padding_value_list must match the number of channels in the tensor.")
    
    half_size = block_size // 2
    
    # Initialize the sub-block with the padding values for each channel
    subblock = torch.stack([torch.full((block_size, block_size), padding_value_list[i]) for i in range(num_channels)])
    
    # Calculate the region within the original tensor that overlaps with the sub-block
    row_start = max(r - half_size, 0)
    row_end = min(r + half_size + 1, height)
    col_start = max(c - half_size, 0)
    col_end = min(c + half_size + 1, width)
    
    # Calculate the region within the sub-block where the original tensor values will be placed
    sub_row_start = half_size - (r - row_start)
    sub_row_end = sub_row_start + (row_end - row_start)
    sub_col_start = half_size - (c - col_start)
    sub_col_end = sub_col_start + (col_end - col_start)
    
    # Place the original tensor values into the appropriate region of the sub-block
    subblock[:, sub_row_start:sub_row_end, sub_col_start:sub_col_end] = tensor[:, row_start:row_end, col_start:col_end]
    
    return subblock

def create_combined_mask(field: torch.Tensor, explosion_map: torch.Tensor) -> torch.Tensor:
    """
    Creates a combined mask that includes information about crates, walls, and current explosion danger zones.

    :param field: A tensor representing the field layout (e.g., walls and crates).
                    -1: Stone wall (indestructible)
                    0: Free space
                    -2: Crate (destructible)
                    5: no bomb countdown
        :param explosion_map: A tensor representing the current explosion danger zones.
                            0: No explosion
                            >0: Explosions or danger (e.g., countdown timers)
        
        :return: A tensor representing the combined mask.
                -1: Inaccessible (walls)
                0: tiles which will explode
                1: Crate (destructible)
    """
    # Initialize the combined mask with the same shape as the field
    combined_mask = torch.ones_like(field, dtype=torch.float32)
    
    # Mark explosion danger zones (non-zero in the explosion map) as inaccessible
    combined_mask[explosion_map != 5] = 0
    
    # Mark stone walls (-1 in the field) as inaccessible in the combined mask
    combined_mask[field == -1] = -1
    
    # Mark crates (1 in the field) in the combined mask
    combined_mask[field == 1] = -2 # earlier crates were also -1.
    
    return combined_mask

# breath first search for finding the next free tile to outrun a bomb explosion.
# mask being the input, start is the agents position
def bfs_find_free_tile(mask, start=(2, 2)):
    if mask[start].item() != 0:### if not in the danger tile, then just return 0 tensor
        return torch.zeros_like(mask, dtype = torch.float32)
    
    if np.random.choice([0,1])==1 :
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([start])
    visited = set()
    visited.add(start)
    parents = {}
    
    while queue:
        current_pos = queue.popleft()
        x, y = current_pos
        
        # Check if the current tile is free
        if mask[x, y].item() == 1:
            output = torch.zeros_like(mask, dtype=torch.float32)
            output[(x, y)] = 1
            temp=2
            while current_pos in parents:
                current_pos = parents[current_pos]
                # output[current_pos] = 1
                output[current_pos]=temp
                temp+=1
            output[start]=0
            return output
        
        # Explore neighboring positions
        for direction in directions:
            new_x, new_y = x + direction[0], y + direction[1]
            
            # Check if the new position is within bounds and not visited
            if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                if (new_x, new_y) not in visited:
                    if mask[new_x, new_y].item() != -1 and mask[new_x, new_y].item()!=-2 :  # Only enqueue if it's not a wall/crate (-1)
                        queue.append((new_x, new_y))
                        visited.add((new_x, new_y))
                        parents[(new_x, new_y)] = (x, y)  # Record the parent
    
    # If no free tile is found, return an empty grid
    return torch.zeros_like(mask, dtype=torch.float32)

def bfs_find_next_crate(mask, start=(2, 2)):
    if mask[start].item() == 0:### if the agent is in the danger tile, then just return 0 tensor
        return torch.zeros_like(mask, dtype = torch.float32)
    
    if np.random.choice([0,1]) == 1:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([start])
    visited = set()
    visited.add(start)
    parents = {}
    
    while queue:
        current_pos = queue.popleft()
        x, y = current_pos
        
        # Check if the current tile is not free
        if mask[x, y].item() == -2:
            output = torch.zeros_like(mask, dtype=torch.float32)
            temp=1
            while current_pos in parents:
                current_pos = parents[current_pos]
                # output[current_pos] = 1
                output[current_pos] = temp
                temp+=1
            output[start] = 0
            return output
        
        # Explore neighboring positions
        for direction in directions:
            new_x, new_y = x + direction[0], y + direction[1]
            
            # Check if the new position is within bounds and not visited
            if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                if (new_x, new_y) not in visited:
                    if mask[new_x, new_y].item() != -1 and mask[new_x, new_y].item() != 0:  # Only enqueue if it's not a wall/crate (-1)
                        queue.append((new_x, new_y))
                        visited.add((new_x, new_y))
                        parents[(new_x, new_y)] = (x, y)  # Record the parent
    
    # If no free tile is found, return an empty grid
    return torch.zeros_like(mask, dtype=torch.float32)



def state_to_features(self,game_state: dict, return_2d_features=False, close_to_crate_check=True) -> torch.Tensor:
    """
    Converts the game state to a multi-channel feature tensor using PyTorch.
    
    :param game_state: A dictionary describing the current game board.
    :return: torch.Tensor representing the feature tensor.
    """
    if game_state is None:
        return None

    field = game_state['field']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    bombs = game_state['bombs']
    self_info = game_state['self']
    others = game_state['others']

    channels = []

    # Field layer: positions our agent can't take
    field_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    field_layer[field == -1] = -1  # Stone walls
    field_layer[field == 1] = 1    # Crates
    ### add opponents as crates
    # Others layer
    # others_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    for _, _, _, (x, y) in others:
        field_layer[x, y] = 1 ### treating opponents as crates
    channels.append(field_layer)

    # Explosion map layer
    bomb_map = torch.ones_like(torch.tensor(field)) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    channels.append(bomb_map)
    # create the combined mask
    comb_mask = create_combined_mask(field=field_layer, explosion_map=bomb_map+explosion_map)
    
    # Coins layer
    coins_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    for coin in coins:
        coins_layer[coin] = 1
    
    ###
    # if not self.train:# we were using it to avoid looping earlier. now we dont need it
    #     if self.recent_states.count(self_info[3]) > 2:
    #         # #print(coins_layer)
    #         coins_layer=torch.bernoulli(torch.full((field.shape), 0.75)) #torch.ones_like(torch.tensor(field), dtype=torch.float32)
    #         # #print("coins_layer for looping case")
    #         # #print(coins_layer)
    channels.append(coins_layer)
    # Bombs layer
    bombs_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    for (x, y), timer in bombs:
        bombs_layer[x, y] = timer
    channels.append(bombs_layer+explosion_map)
    
    # Self layer
    self_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    self_x, self_y = self_info[3]
    self_layer[self_x, self_y] = 1
    # channels.append(self_layer)

    # creating combined mask for bfs search
    ### create subblock for sending mask to bfs search
    subblock_comb_mask = get_subblock_with_padding(tensor = comb_mask.unsqueeze(0),r=self_x,c=self_y,
                                                block_size=BLOCK_SIZE,padding_value_list=[-1])
    subblock_comb_mask = subblock_comb_mask.squeeze(0)
    ### TESTING STH: Avoid going into 0s(dangerous tiles) if agent was not already not in 1
    center = CENTER_POS# Coordinates of the agent (center of the tensor)
    # Check if the agent is in a danger zone (0)
    if subblock_comb_mask[center] != 0:
        # If the agent is not in a danger zone, change all 0s to -1
        subblock_comb_mask[subblock_comb_mask == 0] = -2### what if I change them to fictious crates
    
    ### create the bfs layer now: 1. for nearest crate and the safe location
    bfs_layer_nearest_crate = bfs_find_next_crate(mask = subblock_comb_mask, start=CENTER_POS)# locate the position to bomb
    # to avoid opponent's bombs as well
    subblock_bomb_and_explosion_map = get_subblock_with_padding(
                        tensor = (bomb_map+explosion_map).unsqueeze(0),
                        r=self_x,c=self_y,
                        block_size=BLOCK_SIZE,padding_value_list=[0])
    subblock_bomb_and_explosion_map = subblock_bomb_and_explosion_map.squeeze(0)

    cloned_subblock_bomb_explosion =  subblock_bomb_and_explosion_map.clone()
    cloned_subblock_bomb_explosion[subblock_bomb_and_explosion_map==0]=7
    if subblock_comb_mask[center]!=0:
        subblock_comb_mask[subblock_bomb_and_explosion_map!=5] = -2
    
    #bfs for nearest safe tile
    bfs_layer_safe_location = bfs_find_free_tile(mask = subblock_comb_mask, start=CENTER_POS)# locate the position of nearest safe tile

    ### check if safe path goes in explosion region and avoid it if it does
    # Step 1: Extract the masked tensor
    masked_tensor = bfs_layer_safe_location[cloned_subblock_bomb_explosion == 7]
    # Step 2: Check if any element in the masked tensor is not zero
    if (masked_tensor != 0).any():
        # Step 3: Change the entire bfs_layer_safe_location tensor to zeros
        bfs_layer_safe_location.zero_()  # In-place operation to set all values to zero

    if close_to_crate_check:
        self.close_to_crate = torch.max(bfs_layer_nearest_crate)
    self.close_to_safe_tile = torch.max(bfs_layer_safe_location)
    
    ### finding nearest crate/coin on the overall game field (helps with the case when none is present in the vicinity):
    # create a combined mask for crates and coins
    mask_crate_and_coin = torch.ones_like(torch.tensor(field), dtype=torch.float32)
    mask_crate_and_coin[field == 1]=-2
    mask_crate_and_coin[field == -1]=-1
    mask_crate_and_coin[coins_layer==1]=-2
    recentered_mask_crate_and_coin = get_subblock_with_padding(tensor=mask_crate_and_coin.unsqueeze(0),
                            r=self_x, c=self_y, block_size=field_layer.shape[0]*2 + 1,
                            padding_value_list=[-1]
                            )
    recentered_mask_crate_and_coin = recentered_mask_crate_and_coin.squeeze(0)
    ## use bfs search to find the nearest crate/coin outside the range of visibility
    c = int((field_layer.shape[0]*2+1)/2)
    bfs_crate_and_coin = bfs_find_next_crate(mask = recentered_mask_crate_and_coin, 
                                            start=(c,c))
    bfs_crate_and_coin_sliced = bfs_crate_and_coin[c-2:c+3, c-2:c+3]
    bfs_layer= torch.zeros_like(bfs_crate_and_coin_sliced)
    bfs_layer[bfs_crate_and_coin_sliced!=0]=1 # + bfs_layer_safe_location
    bfs_layer = bfs_layer + bfs_layer_safe_location###will be added as a feature
    # bfs_layer_2 = bfs_layer_nearest_crate + bfs_layer_safe_location# it seems im not using it anywhere.now this is getting saved
    # let's try third thing
    # we will send bfs_layer instead of bfs_layer_2 in the features: THIS IS WORKING REAL WELL
    
    # Stack all channels to form a multi-channel 2D array
    stacked_channels = torch.stack(channels)

    # padding values for the case when agent is at the corner
    padding_value_list=[-1,0,0,0]#
    subblock_info = get_subblock_with_padding(tensor=stacked_channels,
                            r=self_x, c=self_y, block_size=BLOCK_SIZE,
                            padding_value_list=padding_value_list
                            )
    ## checking presence of coins and crates nearby
    if self.train:
        if torch.max(bfs_layer_nearest_crate)==0 and torch.sum(subblock_info[0,:,:]==-2)==0:
            self.no_crate_nearby = 1  
        if  torch.sum(subblock_info[2,:,:])==0:
            self.no_coin_nearby = 1
                
    ####to avoid getting destroyed by its own bomb:
    ## to avoid going back to danger zone created by its own bomb
    subblock_info[0,:,:][subblock_comb_mask==-2]=-2 ### change to fictitous crates instead of walls ### -1 ---> -2
    
    # MAKING AGENT GLIDE AND PLAY SMOOTHLY IN NON-TRAINING MODE
    # OUR NEURAL NETWORK HAS ALREADY LEARNED TO  
    #  1. COLLECT COINS SMOOTHLY (using COINS LAYER FEATURE), 
    #  2. DROP BOMB WHEN NEAR THE CRATES (using WALLS_AND_CRATE Feature layer)
    #  3. AND ESCAPE ITS OWN BOMB(USING BFS LAYER WHICH INCLUDES INFORMATION ABOUT SAFE PATH AS WELL). 
    # NOW, TO FACILITATE SMOOTHER GAMEPLAY AND FURTHER FACILITATE AVOIDING GETTING KILLED by its own bomb
    # WE INTRODUCE FICTITIOUS COINS IN THE COINS FEATURE IN THE DIRECTION (WHICH IS ALSO A PART OF our FEATURES(in bfs_layer)) . 
    # THIS HELPS BECAUSE OUR NEURAL AGENT HAS LEARNED THE COINS FEATURE VERY WELL.
    # (the model works without introducing fictitious coins  as well, but the gameplay is not that smooth).
    if not self.train:
        #removed this from below:"torch.sum(subblock_info[2,:,:])==0 and"
        if  torch.min(bomb_map+explosion_map)==torch.max(bomb_map+explosion_map): # no coins nearby and no emergency: to further facilitate brining close to crate
            subblock_info[2,:,:][bfs_crate_and_coin_sliced!=0]=1
            # if self.recent_states.count(self_info[3]) > 3:
            #     # #print(coins_layer)
            #     subblock_info[2,:,:]=torch.bernoulli(torch.full((subblock_info[2,:,:].shape), 0.75)) #torch.ones_like(torch.tensor(field), dtype=torch.float32)
            #if there are no crate and coins  nearby
        if torch.all(bfs_layer_nearest_crate ==0) and int(torch.sum(subblock_info[0,:,:]==-2)==0) and torch.sum(subblock_info[2,:,:])==0 : #if there are no crate and coins  nearby
            #no crate and coin in the vicinity
            subblock_info[2,:,:][bfs_crate_and_coin_sliced!=0]=1
         
        ### INTRODUCING FICTITOUS COINS IN THE DIRECTION OF SAFE TILE. 
        ### OUR AGENT MOVES IN THAT DIRECTION
        if torch.sum(bfs_layer_safe_location)!=0 and torch.min(bomb_map+explosion_map)!=torch.max(bomb_map+explosion_map):# i.e. if there is no real coin in the vicinity (crates could be present)
            # #print("Immediate priority safe location")
            subblock_info[2,:,:] = torch.zeros_like(subblock_info[2,:,:])
            subblock_info[2,:,:][bfs_layer_safe_location!=0] = 1 #this will also take care of safe tile case

    #add bfs layer into sub-block layers
    # tryinh third thing (saving bfs_layer instead of bfs_layer_2, now it wont be used anywhere, only bfs_layer is being used)
    subblock_info = torch.cat((subblock_info, bfs_layer.unsqueeze(0)), dim=0)

    ### add whole coin and crates info
    if return_2d_features: # if we want to work with image shaped network
        # return stacked_channels
        return subblock_info.float()
    subblock_info = subblock_info.view(-1).float()# THIS IS WHAT WE WILL USE FOR FINAL TOURNAMENT
    return subblock_info

### reduced state, loading the trained autoencoder
# Define your Autoencoder class as before
def state_to_features_encoder(self,game_state: dict, check_close_to_crate=True):

    naive_1d_state= state_to_features(self=self,game_state=game_state,
                                    return_2d_features=self.conv_AE_encoded_features,
                                    close_to_crate_check=check_close_to_crate)
    # naive_1d_state = naive_1d_state.clone().detach().unsqueeze(0).float()
    if self.with_encoder:
        with torch.no_grad():
            if self.conv_AE_encoded_features:
                reduced_feature = self.ae.encoder(naive_1d_state.unsqueeze(0))
            else:
                reduced_feature = self.ae.encoder(naive_1d_state)
            ### NOTE again: I can also include case where we use MLP based encoder. But
            ### since we are not using such an AE, we leave it here for the time being
        
        return reduced_feature.squeeze(0) # update shape is fixed. ### shape is: torch.Size([1, coding_space_dimn])### an extra 1
    else:
        return naive_1d_state.clone().detach()### shape is: torch.Size([naive_feature_dimension])


