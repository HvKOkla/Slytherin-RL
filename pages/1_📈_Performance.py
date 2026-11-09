import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from games.snake.logic import SnakeLogic

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
st.title("Performance Evaluation")

st.markdown("Run multiple episodes to evaluate the statistical distribution of the agent's scores.")


n_episodes = st.slider("Number of Episodes", min_value=10, max_value=1000, value=100, step=10)
model_name = st.text_input("Model Name (without .zip)", "Slytherin_v1")

if st.button("Run Evaluation", use_container_width=True):
    try:
        model = PPO.load(model_name)
        game = SnakeLogic(10)
        scores = []
        
        progress = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_episodes):
            game.reset()
            obs = game.get_observation()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                _, done = game.step(action)
                obs = game.get_observation()
            scores.append(game.score)
            progress.progress((i + 1) / n_episodes)
            status_text.text(f"Episode {i+1}/{n_episodes}")
            
        st.success("Evaluation Complete!")
        
        # Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Score", f"{np.mean(scores):.2f}")
        c2.metric("Max Score", f"{np.max(scores)}")
        c3.metric("Min Score", f"{np.min(scores)}")
        c4.metric("Std Dev", f"{np.std(scores):.2f}")
        
        # Graphique
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(scores, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax.set_title("Score Distribution")
        ax.set_xlabel("Apples Eaten")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")