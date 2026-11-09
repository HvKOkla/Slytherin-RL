import numpy as np
from collections import deque

def get_observation(game):
   
    dx, dy = game.direction
    front, right, left = (dx, dy), (-dy, dx), (dy, -dx)
    head_x, head_y = game.snake[0]
    
    
    danger_front = game._is_danger((head_x + front[0], head_y + front[1]))
    danger_right = game._is_danger((head_x + right[0], head_y + right[1]))
    danger_left  = game._is_danger((head_x + left[0],  head_y + left[1]))

   
    food_dx, food_dy = game.food[0] - head_x, game.food[1] - head_y
    food_right = 1 if (food_dx * right[0] + food_dy * right[1]) > 0 else 0
    food_left  = 1 if (food_dx * left[0] + food_dy * left[1]) > 0 else 0
    food_front = 1 if (food_dx * front[0] + food_dy * front[1]) > 0 else 0

    
    dir_up    = 1 if game.direction == (0, -1) else 0
    dir_down  = 1 if game.direction == (0, 1) else 0
    dir_left  = 1 if game.direction == (-1, 0) else 0
    dir_right = 1 if game.direction == (1, 0) else 0

    
    space_front = _flood_fill(game, (head_x + front[0], head_y + front[1]))
    space_right = _flood_fill(game, (head_x + right[0], head_y + right[1]))
    space_left  = _flood_fill(game, (head_x + left[0],  head_y + left[1]))

    
    return np.array([
        danger_front, danger_right, danger_left, 
        food_front, food_right, food_left, 
        dir_up, dir_down, dir_left, dir_right,
        space_front / 100.0, space_right / 100.0, space_left / 100.0
    ], dtype=np.float32)

def _flood_fill(game, start_pos):
    """Calcule le nombre de cases accessibles depuis start_pos (BFS)."""
    if game._is_danger(start_pos):
        return 0
    
    queue = deque([start_pos])
    visited = {start_pos}
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_pos = (x + dx, y + dy)
            if next_pos not in visited and not game._is_danger(next_pos):
                visited.add(next_pos)
                queue.append(next_pos)
                
    return len(visited)