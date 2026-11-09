# Experiment Log - Slytherin-RL

## [BASELINE] v1.0
- **Hypothesis:** A basic MLP [64, 64] with PPO can learn Snake.
- **Observation:** 10 features (danger, food direction, current direction).
- **Result:** Avg Reward 141.67 ± 56.06 (Max: 318.80).
- **Status:** ✅ FROZEN in `releases/`.

## [H1] v1.1 - Space Awareness (BFS)
- **Hypothesis:** Adding Flood Fill (BFS) to calculate free space in 3 directions will prevent the snake from trapping itself and double the score.
- **Observation:** 13 features (+ free space).
- **Result:** Avg Reward 266.91 ± 73.63 (Max: 519.60).
- **Verdict:** ✅ **CONFIRMED**. The score doubled (+88%). The agent survives much longer.