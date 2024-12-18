import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
import gymnasium as gym
import os
from clearml import Task
import argparse
from ot2_gym_wrapper import OT2Env
from typing_extensions import TypeIs
import tensorflow
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Set the API key for wandb
os.environ['WANDB_API_KEY'] = "76bf2b8cae2c414adb5c3b1292a61d5b3200b733"

# Initialize wandb project
run = wandb.init(project="Task 11", sync_tensorboard=True)

# ClearML task initialization
task = Task.init(
    project_name="Mentor Group K/Group 1/MichonGoddijn",  # Replace YourName with your own name
    task_name="Experiment1",
)

# Set base docker image and queue
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

# Wrap the base environment with your custom wrapper
env = OT2Env()  # Assuming `render` is a parameter in OT2Env

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the PPO model")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the PPO model")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per PPO update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for PPO optimization")
parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor for future rewards")
parser.add_argument("--value_coefficient", type=float, default=0.5, help="Value function loss coefficient")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping range for PPO updates")
parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy architecture to use in PPO")
args, unknown = parser.parse_known_args()  # Handles Jupyter environments gracefully

# Initialize the PPO model with updated hyperparameters
model = PPO(
    policy=args.policy,  # MlpPolicy (JASON)
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,  # 0.0001 (ALEXI)
    batch_size=args.batch_size,        # 32 (MICHON)
    n_steps=args.n_steps,              # 2048 (DAN)
    n_epochs=args.n_epochs,            # 10 (MICHON)
    gamma=args.gamma,                  # 0.98 (ALEXI)
    vf_coef=args.value_coefficient,    # 0.5 (DAN)
    clip_range=args.clip_range,        # 0.2 (JASON)
    tensorboard_log=f"runs/{run.id}",
)

# Create directories for saving models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)

# Create wandb callback
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=model_dir,  # Save models to the specific directory
    verbose=2
)

# Define a custom callback for logging metrics and printing rewards
class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []

    def _on_step(self) -> bool:
        # Collect episode rewards and lengths
        if 'episode' in self.locals:
            episode_info = self.locals['infos'][0].get('episode', {})
            if 'r' in episode_info:  # Episode reward
                self.episode_rewards.append(episode_info['r'])
                wandb.log({"episode_reward": episode_info['r']}, step=self.num_timesteps)
                print(f"Episode Reward: {episode_info['r']}")  # Output reward to console
            if 'l' in episode_info:  # Episode length
                self.episode_lengths.append(episode_info['l'])
                wandb.log({"episode_length": episode_info['l']}, step=self.num_timesteps)

        # Success rate logging (if defined in your environment's info)
        success = self.locals['infos'][0].get('success', None)
        if success is not None:
            self.success_rate.append(success)
            wandb.log({"success_rate": np.mean(self.success_rate)}, step=self.num_timesteps)

        # Log entropy (policy exploration)
        entropy = self.model.logger.name_to_value.get('entropy', None)
        if entropy is not None:
            wandb.log({"entropy": entropy}, step=self.num_timesteps)

        # Log learning rate
        wandb.log({"learning_rate": self.model.learning_rate}, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Log aggregate statistics at the end of training
        wandb.log({
            "average_episode_reward": np.mean(self.episode_rewards),
            "average_episode_length": np.mean(self.episode_lengths),
            "final_success_rate": np.mean(self.success_rate)
        })
        print(f"Training Completed. Final Metrics:")
        print(f"Average Episode Reward: {np.mean(self.episode_rewards)}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths)}")
        print(f"Final Success Rate: {np.mean(self.success_rate)}")

# Add the custom callback to the training process
custom_wandb_callback = CustomWandbCallback()

# Total training timesteps per iteration
time_steps = 6000000

# Training loop
for i in range(10):
    print(f"Starting learn iteration {i + 1}")
    
    # Train the model and log data
    model.learn(
        total_timesteps=time_steps,
        callback=[wandb_callback, custom_wandb_callback],  # Include both callbacks
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    print(f"Completed learn iteration {i + 1}")
    
    # Save the model after each iteration
    model.save(f"{model_dir}/{time_steps * (i + 1)}")
    print(f"Model saved at iteration {i + 1}: {model_dir}/{time_steps * (i + 1)}")