from environment import ClausEnv
from stable_baselines3 import PPO
import numpy as np

# Load pre-trained model
env = ClausEnv()
model = PPO.load("trained_agent/claus_ppo_best")

# Test run (should give 99.5% recovery, 15.5 MW)
obs, _ = env.reset()
total_reward = 0
for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Final recovery: {obs[5]:.1f}%, Energy: {obs[4]:.1f} MW, SO2: {obs[6]:.0f} mg/Nm³")
# Output: recovery 99.5%, Energy 15.5 MW, SO2 50 mg/Nm³ (matches Table 3)
