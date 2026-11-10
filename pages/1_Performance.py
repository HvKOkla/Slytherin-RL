import streamlit as st
import numpy as np
import plotly.express as px 
from engine.agent import GenericEnv
from engine.utils import load_agent
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

st.title("📈 Statistical Performance")
st.markdown("Evaluate the agent across multiple episodes. Metrics are computed on the fly.")


with st.sidebar:
    st.header("Parameters")
    n_episodes = st.slider("Episodes", 10, 500, 100, 10)
    model_path = st.text_input("Model", "releases/v1_1/Slytherin_v1_1")

if st.button("Run Evaluation", type="primary"):
    model = load_agent(model_path)
    env = GenericEnv(SnakeLogic(10), obs_fn=get_obs_function("v1_1"))
    
    scores = []
    progress = st.progress(0, text="Simulating episodes...")
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        scores.append(env.game.score)
        progress.progress((i + 1) / n_episodes)
        
    st.success("Evaluation Complete")
    
   
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Score", f"{np.mean(scores):.1f}")
    c2.metric("Max Score", f"{np.max(scores)}")
    c3.metric("Min Score", f"{np.min(scores)}")
    c4.metric("Consistency (Std)", f"{np.std(scores):.1f}")
    
   
    fig = px.histogram(scores, nbins=20, title="Score Distribution", 
                       labels={'value': 'Apples Eaten'}, color_discrete_sequence=['#2ecc71'])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)