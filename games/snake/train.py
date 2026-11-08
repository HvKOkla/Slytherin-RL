from engine.agent import GenericEnv             # <-- Modifié
from games.snake.logic import SnakeLogic        # <-- Modifié
from config import CONFIG_EDGE                  
from stable_baselines3 import PPO


if __name__ == "__main__":
    print("Démarrage de l'entraînement")
    
    game = SnakeLogic(10)
    env = GenericEnv(game)
    
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
    model.save("Slytherin_v1")
    print("✅ Modèle sauvegardé : Slytherin_v1.zip")