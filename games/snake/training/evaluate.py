import numpy as np
import os
import time # <-- NOUVEAU
from stable_baselines3 import PPO
from engine.agent import GenericEnv
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

def test_model(model_path, obs_version, n_episodes=100):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"❌ ERREUR : Le fichier '{model_path}' n'existe pas.")
        return

    print(f"\n🔍 Évaluation de {model_path} (Obs: {obs_version})...")
    
    game = SnakeLogic(10)
    obs_fn = get_obs_function(obs_version)
    env = GenericEnv(game, obs_fn=obs_fn)
    
    model = PPO.load(model_path)
    
    scores = []
    latencies = [] 
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            start_time = time.perf_counter()
            
            action, _ = model.predict(obs, deterministic=True)
            
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) 
            
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        scores.append(score)
        
    print(f"📊 RÉSULTATS :")
    print(f"Score moyen : {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Score max   : {np.max(scores):.2f}")
    # 📈 Affichage des latences
    print(f"Latence moy : {np.mean(latencies):.3f} ms")
    print(f"Latence p99 : {np.percentile(latencies, 99):.3f} ms")

if __name__ == "__main__":
    print("--- BASELINE (V1.0) ---")
    test_model("releases/v1_0/Slytherin_v1", "v1", n_episodes=100)
    
    print("\n--- NOUVEAU (V1.1) ---")
    test_model("releases/v1_1/Slytherin_v1_1", "v1_1", n_episodes=100)