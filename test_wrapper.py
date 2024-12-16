import gymnasium as gym
import numpy as np
from ot2_gym_wrapper import OT2Env

# Initialize the environment
env = OT2Env(render=True, max_steps=1000)

# Number of episodes to test
num_episodes = 5

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")
        
        step += 1
        done = terminated or truncated
        
        if done:
            print(f"Episode finished after {step} steps.")
            break

env.close()
