import torch
from collections import deque
import numpy as np

def bfs_find_target(mask, start=(2, 2), target_value=1):

    # Check if the starting position is the target and not a danger zone for finding free tiles
    if target_value == 1 and mask[start].item() != 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    
    # Randomly decide the order of directions
    if np.random.choice([0, 1]) == 1:
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
        
        # Check if the current tile is the target
        if mask[x, y].item() == target_value:
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
                    # For free tiles, only enqueue if it's not a wall (-1)
                    # For crates, only enqueue if it's not a wall (-1) or a danger zone (0)
                    if target_value == 1:
                        if mask[new_x, new_y].item() != -1:
                            queue.append((new_x, new_y))
                            visited.add((new_x, new_y))
                            parents[(new_x, new_y)] = (x, y)
                    elif target_value == -2:
                        if mask[new_x, new_y].item() != -1 and mask[new_x, new_y].item() != 0:
                            queue.append((new_x, new_y))
                            visited.add((new_x, new_y))
                            parents[(new_x, new_y)] = (x, y)
    
    # If no target tile is found, return an empty grid
    return torch.zeros_like(mask, dtype=torch.float32)




# two functions:


def bfs_find_free_tile(mask, start=(2, 2)):

    if mask[start].item() != 0:
        return torch.zeros_like(mask, dtype=torch.float32)
    
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
        
        # Check if the current tile is free
        if mask[x, y].item() == 1:
            output = torch.zeros_like(mask, dtype=torch.float32)
            output[(x, y)] = 1
            while current_pos in parents:
                current_pos = parents[current_pos]
                output[current_pos] = 1
            output[start] = 0
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

def bfs_find_next_crate(mask, start=(2, 2)):

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
        
        # Check if the current tile is free
        if mask[x, y].item() == -2:
            output = torch.zeros_like(mask, dtype=torch.float32)
            while current_pos in parents:
                current_pos = parents[current_pos]
                output[current_pos] = 1
            output[start] = 1
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
