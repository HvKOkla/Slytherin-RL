import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GenericEnv(gym.Env):
    def __init__(self, game, obs_fn=None, max_steps=1000):
        super().__init__()
        self.game = game
        self.max_steps = max_steps
        self._steps = 0
        
        # Si aucune fonction n'est fournie, on utilise celle du jeu par défaut
        self.obs_fn = obs_fn if obs_fn is not None else self.game.get_observation
        
        self.action_space = spaces.Discrete(self.game.get_action_space_size())
        
        # On calcule la taille de l'observation
        obs = self.obs_fn(self.game)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self._steps = 0
        return np.array(self.obs_fn(self.game), dtype=np.float32), {}

    def step(self, action):
        self._steps += 1
        reward, done = self.game.step(action)
        truncated = self._steps >= self.max_steps
        obs = np.array(self.obs_fn(self.game), dtype=np.float32)
        return obs, reward, done or truncated, truncated, {}