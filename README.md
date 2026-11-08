
# 🐍 Slytherin-RL v1.0

A lightweight, modular Reinforcement Learning framework designed for Edge AI. 

## 📦 Installation
```bash
git clone https://github.com/HvKOkla/Slytherin-RL.git
cd Slytherin-RL
pip install -r requirements.txt
```

## 🚀 Usage
**Train:** `python -m games.snake.train`
**Watch:** `python -m games.snake.watch` *(requires `Slytherin_v1.zip` from Releases)*

## 🏗️ Architecture
- **`engine/`**: Generic Gymnasium wrapper.
- **`games`**: Game logic and scripts.
- **`config.py`**: Hyperparameters.

## 📊 Specs
- **Architecture:** MLP `[64, 64]` (PPO)
- **Score:** ~20 (Avg)
- **Size:** ~150 KB

## 🗺️ Roadmap
- [x] **v1.0**: Modular Snake Agent.
- [ ] **v1.1**: Pong Environment.
- [ ] **v2.0**: Self-Play.
- [ ] **v3.0**: Generalist Agent.
```