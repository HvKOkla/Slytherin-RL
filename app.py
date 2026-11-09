import streamlit as st

st.set_page_config(page_title="Slytherin-RL | Home", page_icon="🐍", layout="wide")

st.title("🐍 Slytherin-RL | Edge AI Research Dashboard")
st.markdown("""
Welcome to the **Slytherin-RL** dashboard. 

This application allows you to evaluate the performance of our Reinforcement Learning agent trained to master the game of Snake. 

### 🧭 Navigation
Use the sidebar on the left to navigate through the different sections:
- **📈 Performance:** Evaluate the agent's score distribution over multiple episodes.
- **⚡ Speed:** Benchmark the inference latency and throughput (Edge AI focus).
- **🎮 Live Demo:** Watch the v1.1 agent play in real-time.
""")

st.info("Select a page in the sidebar to begin.")

st.markdown("---")
st.markdown("### Model Info (Current Champion: v1.1)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Algorithm:** PPO")
    st.markdown("**Architecture:** MLP `[64, 64]`")
    st.markdown("**Parameters:** 10,053")

with col2:
    st.markdown("**Environment:** Snake (10x10)")
    st.markdown("**Observation:** 13 features (with BFS)")
    st.markdown("**Model Size:** ~150 KB")

with col3:
    st.markdown("**Avg Reward:** 266.91")
    st.markdown("**Max Reward:** 519.60")
    st.markdown("**Latency:** 0.55 ms")