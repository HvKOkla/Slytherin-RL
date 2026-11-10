import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from engine.agent import GenericEnv
from engine.utils import load_agent
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

st.title("⚡ Edge AI Inference Benchmark")
st.markdown("Measure the latency per decision. Crucial for real-time robotics.")

with st.sidebar:
    st.header("⚙️ Parameters")
    n_steps = st.slider("Benchmark Steps", 100, 5000, 1000, 100)
    model_path = st.text_input("Model", "releases/v1_1/Slytherin_v1_1")

if st.button("⏱️ Start Benchmark", type="primary"):
    model = load_agent(model_path)
    env = GenericEnv(SnakeLogic(10), obs_fn=get_obs_function("v1_1"))
    obs, _ = env.reset()
    
    
    for _ in range(50): model.predict(obs, deterministic=True)
        
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        times.append((time.perf_counter() - t0) * 1000)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc: obs, _ = env.reset()
            
    avg_lat = np.mean(times)
    
    # --- KPI ROW ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Latency", f"{avg_lat:.3f} ms")
    c2.metric("P99 Latency", f"{np.percentile(times, 99):.3f} ms")
    c3.metric("Throughput", f"{1000/avg_lat:.0f} FPS")
    
    # Statut (Le petit plus NASA)
    if avg_lat < 1.0:
        st.success("🟢 Real-time capable (< 1ms). Suitable for microcontrollers (TinyML).")
    elif avg_lat < 10.0:
        st.warning("🟡 Standard Edge (< 10ms). Suitable for Raspberry Pi.")
    else:
        st.error("🔴 Too slow for Edge. Needs optimization.")
        
   
    fig = go.Figure(data=go.Scatter(y=times[:200], mode='lines', line=dict(color='#e74c3c')))
    fig.update_layout(title="Latency per Inference (First 200 steps)", 
                      yaxis_title="Time (ms)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)