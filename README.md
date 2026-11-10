
# 🐍 Slytherin-RL | Edge AI Framework

![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Rust](https://img.shields.io/badge/Rust-1.75+-orange)
![SystemVerilog](https://img.shields.io/badge/Hardware-SystemVerilog-blueviolet)

> A lightweight, modular Reinforcement Learning framework designed for Edge AI. 
> *From Python Training to Silicon-Level Inference.*

**[📊 Live Interactive Dashboard](https://slytherin-rl-9dsfymwqfiwkyczznzfuea.streamlit.app/)** *(Test the agent in your browser)*
**[🎮 Human-in-the-Loop Demo](app_live.py)** *(Run `python app_live.py` locally for real-time RLHF)*

---

## 📖 Overview

**Slytherin-RL** is not just a script to play Snake. It is a **modular framework** built to train Reinforcement Learning agents on any environment, optimized for edge deployment.

The project is built on a strict separation of concerns:
- 🧠 **`engine/`**: The generic RL engine (Gymnasium wrapper).
- 🎮 **`games/`**: The specific environment rules (currently Snake).
- 📊 **`app.py`**: A complete Streamlit dashboard for research benchmarking.
- 🎮 **`app_live.py`**: A Gradio interface for Human-in-the-Loop (RLHF) feedback.
- 🦀 **`runtime/`**: An ultra-fast Rust inference engine (ONNX Runtime).
- ⚡ **`hardware/`**: SystemVerilog MAC units for FPGA acceleration.

The current champion, **Slytherin v1.1**, masters the game of Snake (10x10) using a compact `[64, 64]` neural network. It proves that lightweight models can achieve high performance without heavy deep learning frameworks.

---

## ⚡ The Edge Pipeline (Python to Silicon)

We implemented a complete deployment pipeline to demonstrate Edge AI capabilities:

1. **Training:** PyTorch (`[64, 64]` MLP).
2. **Export:** ONNX format (INT8 ready).
3. **Inference (Software):** Rust + ONNX Runtime.
4. **Inference (Hardware):** SystemVerilog MAC array (FPGA).

### 🚀 Latency Benchmark (v1.1)
| Runtime | Latency | Throughput | Environment |
| :--- | :--- | :--- | :--- |
| Python (PyTorch) | 0.55 ms | ~1,800 FPS | CPU (Streamlit) |
| **Rust (ONNX Runtime)** | **0.0205 ms** | **48,882 FPS** | **CPU (Native)** |
| SystemVerilog (Simulated) | ~5 µs | 200,000 FPS | FPGA (Target) |

---

## 📊 Agent Benchmarks (v1.0 vs v1.1)
| Version | Observation Space | Avg Reward | Max Reward | Params |
| :--- | :--- | :--- | :--- | :--- |
| v1.0 | 10 features (Basic) | 141.67 | 318.80 | 10,053 |
| **v1.1** | 13 features (+ BFS) | **266.91** | **519.60** | 10,053 |

---

## 🛠️ Installation & Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/HvKOkla/Slytherin-RL.git
cd Slytherin-RL
```

**2. Setup Python Environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

**3. Usage:**
```bash
# Train an agent from scratch
python -m games.snake.train

# Launch the research dashboard (Streamlit)
streamlit run app.py

# Launch the Human-in-the-Loop interface (Gradio)
python app_live.py
```

**4. Edge Inference (Rust):**
```bash
cd runtime
cargo run --release
```

**5. Hardware Simulation (SystemVerilog):**
```bash
cd hardware
iverilog -g2012 -o sim.out rtl/mac.sv tb/tb_mac.sv
vvp sim.out
```

---

## 🏗️ Project Structure

```text
Slytherin-RL/
├── engine/                 # Generic RL engine
├── games/snake/            # Snake environment (logic, train, watch)
├── app.py                  # Streamlit research dashboard
├── app_live.py             # Gradio RLHF interface
├── runtime/                # Rust inference engine
├── hardware/               # SystemVerilog accelerator
├── config.py               # Centralized hyperparameters
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [x] **v1.0:** Modular Engine + Snake Agent.
- [x] **v1.1:** Improved observation space (BFS spatial awareness).
- [x] **v1.2:** Streamlit Dashboard + Plotly.
- [x] **v1.3:** Human-in-the-Loop (RLHF) with Gradio.
- [x] **v1.4:** Rust ONNX Runtime (0.02 ms latency).
- [x] **Hardware:** SystemVerilog MAC unit (INT8).
- [ ] **v2.0:** Full SystemVerilog MLP Accelerator on FPGA (Tang Nano 9K).
- [ ] **v3.0:** Add a second environment (Pong) with Multi-Task Learning.

---

## 📝 License

This project is open-source and available under the **MIT License**.
```
