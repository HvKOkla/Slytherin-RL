import streamlit as st
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from engine.agent import GenericEnv
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

st.set_page_config(page_title="Live Demo", page_icon="🎮", layout="wide")
st.title("🎮 Live Demo - Slytherin v1.1")
st.markdown("Watch the champion agent (with BFS spatial awareness) play Snake in real-time.")

game = SnakeLogic(10)
obs_fn = get_obs_function("v1_1")
env = GenericEnv(game, obs_fn=obs_fn)

MODEL_PATH = "releases/v1_1/Slytherin_v1_1"
try:
    model = PPO.load(MODEL_PATH)
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

def get_grid_emoji(game):
    grid = []
    for y in range(game.grid_size):
        row = []
        for x in range(game.grid_size):
            if (x, y) == game.snake[0]:
                row.append("🟢")  # Tête
            elif (x, y) in game.snake:
                row.append("🟩")  # Corps
            elif (x, y) == game.food:
                row.append("🔴")  # Pomme
            else:
                row.append("⬛")  # Vide
        grid.append(row)
    return grid

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### 📊 Live Stats")
    score_placeholder = st.empty()
    steps_placeholder = st.empty()
    st.markdown("---")
    st.markdown("**Neural Network Input (13 features):**")
    obs_placeholder = st.empty()

with col1:
    game_placeholder = st.empty()

if st.button("▶️ Start Game", use_container_width=True):
    obs, _ = env.reset()
    done = False
    score = 0
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
        steps += 1
        
        score_placeholder.metric("Reward", f"{score:.1f}")
        steps_placeholder.metric("Steps", steps)
        obs_placeholder.text(f"{np.round(obs, 2)}")
        
        game_placeholder.dataframe(
            pd.DataFrame(get_grid_emoji(game)), 
            use_container_width=True, 
            header=False, 
            index=False
        )
        
        time.sleep(0.1) 

    st.success(f"💀 Game Over! Final Reward: {score:.1f} in {steps} steps.")