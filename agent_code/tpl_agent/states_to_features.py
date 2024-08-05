import numpy as np
import torch

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
    explosion_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    explosion_layer = torch.tensor(explosion_map, dtype=torch.float32)
    channels.append(explosion_layer)

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

    # Others layer
    others_layer = torch.zeros_like(torch.tensor(field), dtype=torch.float32)
    for _, _, _, (x, y) in others:
        others_layer[x, y] = 1
    channels.append(others_layer)

    # Stack all channels to form a multi-channel 2D array
    stacked_channels = torch.stack(channels)

    if return_2d_features: # if we want to work with image shaped network
        return stacked_channels
    
    return stacked_channels.view(-1)
