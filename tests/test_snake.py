import pytest
import numpy as np
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function
from engine.agent import GenericEnv

def test_snake_moves_correctly():
    """Vérifie que le serpent se déplace dans la bonne direction."""
    game = SnakeLogic(10)
    # Le serpent commence au milieu (5,5) et va vers le haut (0, -1)
    assert game.snake[0] == (5, 5)
    
    game.step(0) # Action 0 = Haut
    assert game.snake[0] == (5, 4) # Il doit être monté d'une case

def test_snake_dies_on_wall():
    """Vérifie que le serpent meurt s'il touche un mur."""
    game = SnakeLogic(10)
    game.snake = [(5, 0)] # On le place tout en haut
    game.direction = (0, -1) # Il regarde vers le mur du haut
    
    reward, done = game.step(0) # Il avance dans le mur
    assert done == True
    assert reward == -10 # Pénalité de mort

def test_observation_shape_v1():
    game = SnakeLogic(10)
    obs_fn = get_obs_function("v1")
    obs = obs_fn(game)
    
    assert len(obs) == 10
    assert isinstance(obs, np.ndarray)

def test_observation_shape_v1_1():
    game = SnakeLogic(10)
    obs_fn = get_obs_function("v1_1")
    obs = obs_fn(game)
    
    assert len(obs) == 13
    assert isinstance(obs, np.ndarray)

def test_generic_env_reset_and_step():

    game = SnakeLogic(10)
    obs_fn = get_obs_function("v1_1")
    env = GenericEnv(game, obs_fn=obs_fn)
    
    
    obs, info = env.reset()
    assert len(obs) == 13
    
    
    action = env.action_space.sample() # Prend une action au hasard
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert len(obs) == 13
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)