import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from stable_baselines3 import PPO
from games.snake.logic import SnakeLogic

st.title("⚡ Edge AI Inference Speed")
st.markdown("Measure how fast the agent can make a decision. Crucial for robotics.")

model_name = st.text_input("Model Name (without .zip)", "Slytherin_v1")
n_bench_steps = st.slider("Number of Inference Steps", min_value=100, max_value=5000, value=1000, step=100)

if st.button("⏱️ Measure Inference Speed", use_container_width=True):
    try:
        model = PPO.load(model_name)
        game = SnakeLogic(10)
        game.reset()
        
        dummy_obs = game.get_observation()
        for _ in range(100):
            model.predict(dummy_obs, deterministic=True)
            
        obs = game.get_observation()
        times = []
        
        start_total = time.perf_counter()
        for _ in range(n_bench_steps):
            t0 = time.perf_counter()
            model.predict(obs, deterministic=True)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000) 
        end_total = time.perf_counter()
        
        total_time = end_total - start_total
        avg_latency = np.mean(times)
        p99_latency = np.percentile(times, 99)
        fps = n_bench_steps / total_time
        
        st.success("Benchmark Complete!")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Latency", f"{avg_latency:.3f} ms")
        c2.metric("99th Percentile", f"{p99_latency:.3f} ms")
        c3.metric("Throughput", f"{fps:.0f} FPS")
        c4.metric("Device", "CUDA" if torch.cuda.is_available() else "CPU")
        
        if avg_latency < 1.0:
            st.success("Real-time capable (< 1ms). Can run on microcontrollers.")
        elif avg_latency < 10.0:
            st.warning("Standard Edge (< 10ms). Suitable for Raspberry Pi.")
        else:
            st.error("Too slow for Edge. Needs optimization.")
            
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(times[:200], color='#e74c3c', alpha=0.8)
        ax.set_title("Latency per Inference (First 200 steps)")
        ax.set_ylabel("Time (ms)")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {e}")