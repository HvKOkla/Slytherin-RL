from engine.agent import GenericEnv             
from games.snake.core.logic import SnakeLogic   
from games.snake.envs import get_obs_function 
from config import CONFIG_EDGE                  
from stable_baselines3 import PPO
import os
from datetime import datetime

if __name__ == "__main__":
    OBS_VERSION = "v1_1" 
    
    print(f"Démarrage de l'entraînement avec l'observation : {OBS_VERSION}")
    
    game = SnakeLogic(10)
    obs_fn = get_obs_function(OBS_VERSION)
    env = GenericEnv(game, obs_fn=obs_fn)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        seed=CONFIG_EDGE["seed"],
        policy_kwargs=dict(net_arch=CONFIG_EDGE["net_arch"]),
        n_steps=CONFIG_EDGE["n_steps"], 
        batch_size=CONFIG_EDGE["batch_size"],
        learning_rate=CONFIG_EDGE["learning_rate"], 
        gamma=CONFIG_EDGE["gamma"],
        ent_coef=CONFIG_EDGE["ent_coef"],
    )
    
    model.learn(total_timesteps=CONFIG_EDGE["total_timesteps"])
    
    
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    model_name = f"models/Slytherin_{OBS_VERSION}_{timestamp}"
    
    model.save(model_name)
    print(f"✅ Modèle sauvegardé en toute sécurité : {model_name}.zip")