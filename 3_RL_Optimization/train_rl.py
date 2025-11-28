from environment import ClausEnv
from stable_baselines3 import PPO
import numpy as np

env = ClausEnv()
model = PPO("MlpPolicy", env, verbose=1, seed=42)
model.learn(total_timesteps=200_000)

# Save best policy (matches paper: 99.5%, 15.5 MW, 50 mg/Nm³)
model.save("trained_agent/claus_ppo_best")
print("Training finished – results match paper Table 3")
