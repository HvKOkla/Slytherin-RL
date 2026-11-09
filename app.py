import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from stable_baselines3 import PPO
import torch 

from games.snake.logic import SnakeLogic


st.set_page_config(
    page_title="Slytherin-RL | Research Dashboard",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐍 Slytherin-RL | Edge AI Research Dashboard")
st.markdown("A modular Reinforcement Learning framework for Edge AI. Evaluate performance, behavior, and inference speed.")


st.sidebar.header("Evaluation Parameters")
model_name = st.sidebar.text_input("Model Name (without .zip)", "Slytherin_v1")
n_episodes = st.sidebar.slider("Number of Episodes (Score)", 10, 100, 20)
n_bench_steps = st.sidebar.slider("Number of Inference Steps (Speed)", 100, 5000, 1000)


tab1, tab2, tab3, tab4 = st.tabs([
    "Performance", 
    "Edge Inference Speed", 
    "Architecture", 
    "Specs"
])


with tab1:
    st.header("Performance Evaluation")
    if st.button("🚀 Evaluate Score", use_container_width=True):
        try:
            model = PPO.load(model_name)
            game = SnakeLogic(10)
            scores = []
            
            progress = st.progress(0)
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
            
            st.success("Evaluation Complete!")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average Score", f"{np.mean(scores):.2f}")
            c2.metric("Max Score", f"{np.max(scores)}")
            c3.metric("Min Score", f"{np.min(scores)}")
            c4.metric("Std Dev", f"{np.std(scores):.2f}")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(scores, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
            ax.set_title("Score Distribution")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")
            
with tab2:
    st.header("Edge AI Inference Speed (Latency)")
    st.markdown("Measure how fast the agent can make a decision. Crucial for robotics and microcontrollers.")
    
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
            
            with st.spinner(f"Measuring {n_bench_steps} forward passes..."):
                start_total = time.perf_counter()
                for _ in range(n_bench_steps):
                    t0 = time.perf_counter()
                    model.predict(obs, deterministic=True) # C'est ici que l'IA "réfléchit"
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000) # Convertir en millisecondes
                end_total = time.perf_counter()
            
            # Calculs statistiques
            total_time = end_total - start_total
            avg_latency = np.mean(times)
            p99_latency = np.percentile(times, 99) # Le pire cas (99% des requêtes sont plus rapides)
            fps = n_bench_steps / total_time
            
            st.success("Benchmark Complete!")
            
            st.markdown("### 📊 Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average Latency", f"{avg_latency:.3f} ms")
            col2.metric("99th Percentile (p99)", f"{p99_latency:.3f} ms", help="Worst-case scenario for real-time applications")
            col3.metric("Throughput (FPS)", f"{fps:.0f}")
            col4.metric("Device", "CUDA" if torch.cuda.is_available() else "CPU")
            
            st.markdown("---")
            st.markdown("### Edge AI Context")
            if avg_latency < 1.0:
                st.success("**Real-time capable**: Latency < 1ms. This model can run on a microcontroleur (ESP32) or high-speed robotic control loop.")
            elif avg_latency < 10.0:
                st.warning("**Standard Edge**: Latency < 10ms. Suitable for Raspberry Pi, smartphones, and standard IoT devices.")
            else:
                st.error("**Too slow for Edge**: Latency > 10ms. Needs optimization (ONNX export, INT8 Quantization) for real-time applications.")
                
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(times[:200], color='#e74c3c', alpha=0.8) 
            ax.set_title("Latency per Inference (First 200 steps)")
            ax.set_ylabel("Time (ms)")
            ax.set_xlabel("Inference Step")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error during benchmark: {e}")


with tab3:
    st.header("Neural Network Architecture")
    st.code("""
Policy: MlpPolicy
Architecture: [64, 64]
Activation: Tanh
Params: ~5,000
    """, language="yaml")


with tab4:
    st.header("Environment & Model Specs")
    st.markdown("""
    - **Game:** Snake (10x10)
    - **State Space:** Box(10,)
    - **Action Space:** Discrete(4)
    - **Model Size:** ~150 KB (PyTorch)
    """)