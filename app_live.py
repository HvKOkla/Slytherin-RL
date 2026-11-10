import gradio as gr
import pandas as pd
from stable_baselines3 import PPO

from engine.agent import GenericEnv
from engine.human_wrapper import HumanFeedbackWrapper
from games.snake.core.logic import SnakeLogic
from games.snake.envs import get_obs_function

# --- INITIALIZATION ---
MODEL_PATH = "releases/v1_1/Slytherin_v1_1"
model = PPO.load(MODEL_PATH)
obs_fn = get_obs_function("v1_1")

# Create environment and wrap it
base_env = GenericEnv(SnakeLogic(10), obs_fn=obs_fn)
env = HumanFeedbackWrapper(base_env, human_weight=0.5)

# Game state dictionary
state = {
    "obs": None,
    "done": False,
    "reward": 0.0,
    "steps": 0
}

def get_grid(game):
    grid = []
    for y in range(game.grid_size):
        row = []
        for x in range(game.grid_size):
            if (x, y) == game.snake[0]: row.append("🟢")  # Head
            elif (x, y) in game.snake: row.append("🟩")  # Body
            elif (x, y) == game.food: row.append("🔴")  # Apple
            else: row.append("⬛")  # Empty
        grid.append(row)
    return pd.DataFrame(grid)

def reset_game():
    state["obs"], _ = env.reset()
    state["done"] = False
    state["reward"] = 0.0
    state["steps"] = 0
    # On utilise env.env.game pour accéder au jeu à l'intérieur du wrapper
    return get_grid(env.env.game), "Reward: 0.0 | Steps: 0"

def step_game():
    if state["done"]:
        return get_grid(env.env.game), f"💀 Game Over! Final Reward: {state['reward']:.1f}"
    
    # Agent predicts action
    action, _ = model.predict(state["obs"], deterministic=True)
    
    # Environment steps (taking human feedback into account)
    state["obs"], reward, term, trunc, _ = env.step(action)
    state["reward"] += reward
    state["steps"] += 1
    state["done"] = term or trunc
    
    return get_grid(env.env.game), f"Reward: {state['reward']:.1f} | Steps: {state['steps']}"

def feedback_positive():
    env.set_human_feedback(1)
    return "👍 Positive feedback registered!"

def feedback_negative():
    env.set_human_feedback(-1)
    return "👎 Negative feedback registered!"
    
with gr.Blocks(title="Slytherin-RL | Human-in-the-Loop") as demo:
    gr.Markdown("# 🐍 Slytherin-RL : Human-in-the-Loop (RLHF)")
    gr.Markdown("Step through the game and guide the agent using the 👍/👎 buttons. Its reward signal will be updated instantly.")
    
    with gr.Row():
        with gr.Column(scale=2):
            game_view = gr.Dataframe(value=reset_game()[0], label="Game Board", interactive=False)
            with gr.Row():
                btn_step = gr.Button("⏭️ Step Forward", variant="primary")
                btn_reset = gr.Button("🔄 Reset Game", variant="secondary")
            
        with gr.Column(scale=1):
            stats = gr.Textbox(value="Reward: 0.0 | Steps: 0", label="Live Stats", interactive=False)
            gr.Markdown("### 🧠 Guide the Agent")
            btn_good = gr.Button("👍 Good Action (+1)", variant="success")
            btn_bad = gr.Button("👎 Bad Action (-1)", variant="stop")
            feedback_text = gr.Textbox(label="Last Feedback Status", interactive=False)

    # Bind buttons to functions
    btn_step.click(fn=step_game, outputs=[game_view, stats])
    btn_reset.click(fn=reset_game, outputs=[game_view, stats])
    btn_good.click(fn=feedback_positive, outputs=[feedback_text])
    btn_bad.click(fn=feedback_negative, outputs=[feedback_text])

if __name__ == "__main__":
    demo.launch()