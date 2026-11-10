import streamlit as st
import time
import pandas as pd
from engine.agent import GenericEnv
from engine.utils import load_agent
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

st.title("🎮 Live Demo")
st.markdown("Watch the agent play in real-time.")

# Initialisation de l'état (si ce n'est pas déjà fait)
if "game" not in st.session_state:
    st.session_state.game = SnakeLogic(10)
    st.session_state.env = GenericEnv(st.session_state.game, obs_fn=get_obs_function("v1_1"))
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.score = 0

def get_grid(game):
    return [[ "🟢" if (x,y)==game.snake[0] else "🟩" if (x,y) in game.snake 
              else "🔴" if (x,y)==game.food else "⬛" for x in range(game.grid_size)] 
             for y in range(game.grid_size)]

# Layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### 📊 Live Stats")
    score_ph = st.empty()
    obs_ph = st.empty()
    if st.button("🔄 Reset Game"):
        st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.score = 0
        st.rerun()

with col1:
    game_ph = st.empty()

if not st.session_state.done:
    model = load_agent("releases/v1_1/Slytherin_v1_1")
    action, _ = model.predict(st.session_state.obs, deterministic=True)
    st.session_state.obs, reward, term, trunc, _ = st.session_state.env.step(action)
    
    if term or trunc:
        st.session_state.done = True
    else:
        st.session_state.score += reward
        
    score_ph.metric("Score", f"{st.session_state.score:.1f}")
    obs_ph.text(f"Input: {st.session_state.obs}")
    game_ph.dataframe(pd.DataFrame(get_grid(st.session_state.game)), header=False, index=False)
    
    time.sleep(0.1)
    st.rerun() # Relance le script pour le prochain pas !
else:
    st.error(f"💀 Game Over! Final Score: {st.session_state.score:.1f}")