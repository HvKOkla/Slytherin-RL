import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from engine.agent import GenericEnv
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

st.title("📈 Performance Evaluation")
st.markdown("Run multiple episodes to evaluate the statistical distribution of the agent's scores.")

n_episodes = st.slider("Number of Episodes", min_value=10, max_value=1000, value=100, step=10)
model_name = st.text_input("Model Path (without .zip)", "releases/v1_1/Slytherin_v1_1")
obs_version = st.text_input("Observation Version", "v1_1")

if st.button("🚀 Run Evaluation", use_container_width=True):
    try:
        # 1. Chargement du modèle et de l'environnement
        model = PPO.load(model_name)
        game = SnakeLogic(10)
        env = GenericEnv(game, obs_fn=get_obs_function(obs_version))
        
        scores = []
        progress = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            # On sauvegarde le nombre de pommes mangées (le vrai score du jeu)
            scores.append(game.score) 
            progress.progress((i + 1) / n_episodes)
            status_text.text(f"Episode {i+1}/{n_episodes}")
            
        st.success("Evaluation Complete!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Apples", f"{np.mean(scores):.2f}")
        c2.metric("Max Apples", f"{np.max(scores)}")
        c3.metric("Min Apples", f"{np.min(scores)}")
        c4.metric("Std Dev", f"{np.std(scores):.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(scores, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax.set_title("Score Distribution")
        ax.set_xlabel("Apples Eaten")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")