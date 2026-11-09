import numpy as np
import os
from stable_baselines3 import PPO
from engine.agent import GenericEnv
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

def test_model(model_path, obs_version, n_episodes=100):
    # On vérifie si le fichier existe (avec ou sans .zip)
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"❌ ERREUR : Le fichier '{model_path}' n'existe pas. On passe au suivant.")
        return

    print(f"Évaluation de {model_path} (Obs: {obs_version})...")
    
    game = SnakeLogic(10)
    obs_fn = get_obs_function(obs_version)
    env = GenericEnv(game, obs_fn=obs_fn)
    
    model = PPO.load(model_path)
    
    scores = []
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        scores.append(score)
        
    print(f"📊 RÉSULTATS :")
    print(f"Score moyen : {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Score max   : {np.max(scores):.2f}")

if __name__ == "__main__":
    print("--- BASELINE (V1.0)")
    test_model("releases/Slytherin_v1", "v1", n_episodes=100)
    
    print("NOUVEAU (V1.1)")
    test_model("models/Slytherin_v1_1", "v1_1", n_episodes=100)