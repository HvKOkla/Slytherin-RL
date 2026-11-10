import gymnasium as gym

class HumanFeedbackWrapper(gym.Wrapper):
    def __init__(self, env, human_weight=0.5):
        super().__init__(env)
        self.human_weight = human_weight
        self.human_feedback = 0  # 0 = neutral, 1 = good, -1 = bad

    def step(self, action):
        # 1. Environment steps normally
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Add human feedback to the reward
        total_reward = reward + (self.human_feedback * self.human_weight)
        
        # 3. Reset feedback (it only applies to the current step)
        self.human_feedback = 0 
        
        return obs, total_reward, terminated, truncated, info

    def set_human_feedback(self, feedback):
        self.human_feedback = feedback