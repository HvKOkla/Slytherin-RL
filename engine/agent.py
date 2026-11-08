import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GenericEnv(gym.Env):
    def __init__(self, game, max_steps=1000):
        super().__init__()
        self.game = game
        self.max_steps = max_steps
        self._steps = 0
        self.action_space = spaces.Discrete(self.game.get_action_space_size())
        obs = self.game.get_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self._steps = 0
        return np.array(self.game.get_observation(), dtype=np.float32), {}

    def step(self, action):
        self._steps += 1
        reward, done = self.game.step(action)
        truncated = self._steps >= self.max_steps
        obs = np.array(self.game.get_observation(), dtype=np.float32)
        return obs, reward, done or truncated, truncated, {}