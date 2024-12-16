import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.sim = Simulation(num_agents=1, render=render)
        
        # Action space: control velocities for x, y, z axes
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.goal_position = np.random.uniform(low=[-0.5, -0.5, 0.0], high=[0.5, 0.5, 0.5])
        state = self.sim.reset(num_agents=1)
        
        # Dynamically fetch the robot ID key
        robot_id_key = list(state.keys())[0]  # Get the first robot ID from the state
        pipette_position = state[robot_id_key]["pipette_position"]
        observation = np.array(pipette_position + list(self.goal_position), dtype=np.float32)
        self.steps = 0
        return observation, {}



    def step(self, action):
        action = np.clip(action, -1.0, 1.0)  # Clip actions to valid range
        scaled_action = list(action) + [0]  # Append '0' for no liquid dispensing
        state = self.sim.run([scaled_action])
    
        # Dynamically fetch the robot ID key
        robot_id_key = list(state.keys())[0]  # Get the first robot ID from the state
        pipette_position = state[robot_id_key]["pipette_position"]
        observation = np.array(pipette_position + list(self.goal_position), dtype=np.float32)

        # Reward: negative distance to the goal
        distance = np.linalg.norm(np.array(pipette_position) - np.array(self.goal_position))
        reward = -distance

        # Termination: goal reached within a threshold
        terminated = distance < 0.01

        # Truncation: maximum steps exceeded
        truncated = self.steps >= self.max_steps

        info = {}
        self.steps += 1

        return observation, reward, terminated, truncated, info


    def render(self, mode="human"):
        pass  # Rendering is handled by the simulation's GUI if enabled

    def close(self):
        self.sim.close()
