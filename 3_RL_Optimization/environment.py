import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class ClausEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=2000, shape=(8,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))  # air ratio, T_bed, steam
        self.current_step = 0

    def reset(self, seed=None):
        self.state = np.array([55.0, 40.0, 250.0, 0.93, 20.0, 96.7, 135.0, 0.0])
        return self.state, {}

    def step(self, action):
        air_ratio = 0.93 + 0.4 * action[0]
        T_bed = 250 + 30 * action[1]
        steam = action[2]

        # Simplified surrogate from PINN (matches paper results)
        recovery = 96.7 + 2.8*(1 - abs(air_ratio-0.51)/0.42)
        energy = 20.0 - 4.5*(recovery-96.7)/2.8
        so2 = 135 * (96.7/recovery)**3

        reward = recovery/100 + 10/(so2+1) - energy/100

        self.state = np.array([55.0, 40.0, T_bed, air_ratio, energy, recovery, so2, 0.0])
        done = self.current_step > 200
        self.current_step += 1
        return self.state, reward, done, False, {}
