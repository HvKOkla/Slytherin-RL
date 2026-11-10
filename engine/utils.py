import streamlit as st
from stable_baselines3 import PPO

@st.cache_resource
def load_agent(model_path):
    return PPO.load(model_path)