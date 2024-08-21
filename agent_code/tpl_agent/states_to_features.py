import numpy as np
import torch
import heapq
from collections import deque
import torch.nn as nn
### this code implements the following:
# 1. naive full game information
# 2. reduced state representation using AE (encoder to be precise)

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
                    1: Crate (destructible)
        :param explosion_map: A tensor representing the current explosion danger zones.
                            0: No explosion
                            >0: Explosions or danger (e.g., countdown timers)
        
        :return: A tensor representing the combined mask.
                -1: Inaccessible (wall or explosion)
                0: Accessible (free space)
                1: Crate (destructible)
    """
    # Initialize the combined mask with the same shape as the field
    combined_mask = torch.ones_like(field, dtype=torch.float32)

    # Mark stone walls (-1 in the field) as inaccessible in the combined mask
    combined_mask[field == -1] = -1
    
    # Mark crates (1 in the field) in the combined mask
    combined_mask[field == 1] = -1
    
    # Mark explosion danger zones (non-zero in the explosion map) as inaccessible
    combined_mask[explosion_map != 5] = 0
    
    return combined_mask


def state_to_features(game_state: dict, return_2d_features=False) -> torch.Tensor:
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
    channels.append(field_layer)

    # Explosion map layer
    bomb_map = torch.ones_like(torch.tensor(field)) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    channels.append(bomb_map)
    # create the combined mask
    comb_mask = create_combined_mask(field=field_layer, explosion_map=bomb_map)

    # Coins layer
    coins_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    for coin in coins:
        coins_layer[coin] = 1
    channels.append(coins_layer)
    # Bombs layer
    bombs_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    for (x, y), timer in bombs:
        bombs_layer[x, y] = timer
    channels.append(bombs_layer)

    # Self layer
    self_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    self_x, self_y = self_info[3]
    self_layer[self_x, self_y] = 1
    channels.append(self_layer)

    # creating combined mask for bfs search
    ### create subblock for sending mask to bfs search
    subblock_comb_mask = get_subblock_with_padding(tensor = comb_mask.unsqueeze(0),r=self_x,c=self_y,
                                                block_size=5,padding_value_list=[-1])
    subblock_comb_mask = subblock_comb_mask.squeeze(0)
    # create the bfs layer now
    bfs_layer = bfs_find_free_tile(mask = subblock_comb_mask, start=(2,2))# later start would be (3,3)

    # Others layer
    # others_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    # for _, _, _, (x, y) in others:
    #     others_layer[x, y] = 1
    # channels.append(others_layer)

    # Stack all channels to form a multi-channel 2D array
    stacked_channels = torch.stack(channels)

    # padding values for the case when agent is at the corner
    ### I am not quite sure about padding of 5 we used earlier.
    # field, explosion map, coins, 
    # bombs-position with countdown at that point,
    #self layer
    padding_value_list=[-1,0,0,0,0]#[-1,-1,0,5,0]#,0] ### an update: for combined mask layer, I need to update
    subblock_info = get_subblock_with_padding(tensor=stacked_channels,
                            r=self_x, c=self_y, block_size=5,
                            padding_value_list=padding_value_list
                            )
    # add bfs layer into sub-block layer
    subblock_info = torch.cat((subblock_info, bfs_layer.unsqueeze(0))
                            , dim=0)
    # print("_"*10)
    # print("let's check new sub-block info is:")
    # print(subblock_info)
    if return_2d_features: # if we want to work with image shaped network
        # return stacked_channels
        return subblock_info.float()
    
    # return stacked_channels.view(-1)
    return subblock_info.view(-1).float()

# breath first search for finding the next free tile to outrun a bomb explosion.
# mask being the input, start is the agents position
from collections import deque
import torch

def bfs_find_free_tile(mask, start=(2, 2)):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
            while current_pos in parents:
                current_pos = parents[current_pos]
                output[current_pos] = 1
            return output
        
        # Explore neighboring positions
        for direction in directions:
            new_x, new_y = x + direction[0], y + direction[1]
            
            # Check if the new position is within bounds and not visited
            if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                if (new_x, new_y) not in visited:
                    if mask[new_x, new_y].item() != -1:  # Only enqueue if it's not a wall/crate (-1)
                        queue.append((new_x, new_y))
                        visited.add((new_x, new_y))
                        parents[(new_x, new_y)] = (x, y)  # Record the parent
    
    # If no free tile is found, return an empty grid
    return torch.zeros_like(mask, dtype=torch.float32)

# def bfs_find_free_tile(mask, start=(2, 2)):
#     # Directions for moving in the grid (up, down, left, right)
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
#     # Initialize the queue with the starting position
#     queue = deque([start])


#     # Initialize a set for visited positions
#     visited = set()
#     visited.add(start)
#     # so we can do backtracking to find the path
#     parents = {}
    
#     while queue:
#         current_pos = queue.popleft()
#         x, y = current_pos
        
#         # Check if the current tile is free
#         if mask[x, y].item() == 1:
#             output = torch.zeros_like(mask, dtype=torch.float32)
#             # Mark the first free tile 
#             output[(x, y)] = 1
#             # Backtrack from the free tile to the start using the parents dictionary
#             while current_pos in parents:
#                 output[current_pos] = 1
#                 current_pos = parents[current_pos]
#             return output
        
#         # Explore neighboring positions
#         for direction in directions:
#             new_x, new_y = x + direction[0], y + direction[1]
            
#             # Check if the new position is not visited
#             if (new_x, new_y) not in visited:
#                 if mask[new_x, new_y].item() != -1:  # Only enqueue if it's not a wall/crate (-1)
#                     queue.append((new_x, new_y))
#                     visited.add((new_x, new_y))
#                     parents[(new_x, new_y)] = (x, y)  # Record the parent
    
#     # If no free tile is found, return None or handle it appropriately
#     return torch.zeros_like(mask, dtype=torch.float32)

### reduced state, loading the trained autoencoder
# Define your Autoencoder class as before
def state_to_features_encoder(self,game_state: dict):

    naive_1d_state= state_to_features(game_state=game_state,
                                    return_2d_features=self.conv_AE_encoded_features)
    # print(f"naive_1d_state.shape: {naive_1d_state.shape}")
    # naive_1d_state = naive_1d_state.clone().detach().unsqueeze(0).float()
    if self.with_encoder:
        with torch.no_grad():
            if self.conv_AE_encoded_features:
                reduced_feature = self.ae.encoder(naive_1d_state.unsqueeze(0))
            else:
                reduced_feature = self.ae.encoder(naive_1d_state)
            ### NOTE again: I can also include case where we use MLP based encoder. But
            ### since we are not using such an AE, we leave it here for the time being
        # print(f"shape of the reduced features: {reduced_feature.shape}")
        return reduced_feature.squeeze(0) # update shape is fixed. ### shape is: torch.Size([1, coding_space_dimn])### an extra 1
    else:
        # print(f"shape of the naive features: {naive_1d_state.clone().detach().shape}")
        return naive_1d_state.clone().detach()### shape is: torch.Size([naive_feature_dimension])


