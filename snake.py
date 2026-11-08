import random

class SnakeLogic:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, -1)
        self.food = self._spawn_food()
        self.score = 0
        self.done = False

    def _spawn_food(self):
        free = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.snake]
        if not free:
            self.done = True
            return self.snake[0]
        return random.choice(free)

    def _is_danger(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or pos in self.snake:
            return 1
        return 0

    def step(self, action):
        if action == 0 and self.direction != (0, 1): self.direction = (0, -1)
        elif action == 1 and self.direction != (0, -1): self.direction = (0, 1)
        elif action == 2 and self.direction != (1, 0): self.direction = (-1, 0)
        elif action == 3 and self.direction != (-1, 0): self.direction = (1, 0)

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            return -10, self.done

        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        self.snake.insert(0, new_head)
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            reward = 10
        else:
            self.snake.pop()
            reward = 0.1 if new_dist < old_dist else (-0.1 if new_dist > old_dist else 0)

        return reward, self.done

    def get_action_space_size(self): 
        return 4

    def get_observation(self):
        dx, dy = self.direction
        front, right, left = (dx, dy), (-dy, dx), (dy, -dx)
        head_x, head_y = self.snake[0]
        
        danger_front = self._is_danger((head_x + front[0], head_y + front[1]))
        danger_right = self._is_danger((head_x + right[0], head_y + right[1]))
        danger_left  = self._is_danger((head_x + left[0],  head_y + left[1]))

        food_dx, food_dy = self.food[0] - head_x, self.food[1] - head_y
        food_right = 1 if (food_dx * right[0] + food_dy * right[1]) > 0 else 0
        food_left  = 1 if (food_dx * left[0] + food_dy * left[1]) > 0 else 0
        food_front = 1 if (food_dx * front[0] + food_dy * front[1]) > 0 else 0

        dir_up    = 1 if self.direction == (0, -1) else 0
        dir_down  = 1 if self.direction == (0, 1) else 0
        dir_left  = 1 if self.direction == (-1, 0) else 0
        dir_right = 1 if self.direction == (1, 0) else 0

        return [danger_front, danger_right, danger_left, food_front, food_right, food_left, dir_up, dir_down, dir_left, dir_right]