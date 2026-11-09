# 🐍 Slytherin-RL v1.0

![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

> A lightweight, modular Reinforcement Learning framework designed for Edge AI. 

**[Live Interactive Dashboard](https://slytherin-rl-9dsfymwqfiwkyczznzfuea.streamlit.app/)** *(Click to test the agent in your browser)*

## 📖 Overview

**Slytherin-RL** is not just a script to play Snake. It is a **modular framework** built to train Reinforcement Learning agents on any environment. 

The project is built on a strict separation of concerns:
- 🧠 **`engine/`**: The generic RL engine (Gymnasium wrapper). Write once, use for any game.
- 🎮 **`games/`**: The specific environment rules (currently Snake).
- 📊 **`app.py`**: A complete Streamlit dashboard for research benchmarking.

The first agent, **Slytherin v1.0**, masters the game of Snake (10x10). It uses a compact `[64, 64]` neural network, proving that lightweight models can achieve high performance without heavy deep learning frameworks.



## 📊 Benchmarks (v1.1)
| Version | Observation Space | Avg Reward | Max Reward | Latency | Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0** | 10 features (Basic) | 141.67 | 318.80 | 0.82 ms | 10,053 |
| **v1.1** | 13 features (+ BFS) | **266.91** | **519.60** | ~0.90 ms | 10,053 |


## Installation & Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/HvKOkla/Slytherin-RL.git
cd Slytherin-RL
```

**2. Create a virtual environment and install:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**3. Usage:**
```bash
# Train an agent from scratch
python -m games.snake.train

# Watch the agent play (requires Slytherin_v1.zip from Releases)
python -m games.snake.watch

# Launch the research dashboard
streamlit run app.py
```

---

## 🏗️ Project Structure

```text
Slytherin-RL/
├── engine/                 # Generic RL engine (Gymnasium wrapper)
├── games/snake/            # Snake environment (logic, train, watch)
├── app.py                  # Streamlit research dashboard
├── config.py               # Centralized hyperparameters
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [x] **v1.0 (Current):** Modular Engine + Snake Agent + Streamlit Dashboard.
- [ ] **v1.1:** Improve Snake observation space (vision of walls).
- [ ] **v1.2:** Add a second environment (Pong).
- [ ] **v2.0:** Implement Self-Play (2-player Snake).
- [ ] **v3.0:** Multi-Task Learning setup to train a single "Generalist" model.
- [ ] **Future:** ONNX export, INT8 Quantization, and Robotic Arm simulation.

---

## 📝 License

This project is open-source and available under the **MIT License**.
```