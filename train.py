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
    project_name="Mentor Group K/Group 1/MichonGoddijn",
    task_name="Experiment1",
)

# Set base docker image and queue
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

# Wrap the base environment with your custom wrapper
env = OT2Env()

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
args, unknown = parser.parse_known_args()

# Initialize the PPO model with updated hyperparameters
model = PPO(
    policy=args.policy,
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    vf_coef=args.value_coefficient,
    clip_range=args.clip_range,
    tensorboard_log=f"runs/{run.id}",
)

# Create directories for saving models
model_dir = f"models/{run.id}"
os.makedirs(model_dir, exist_ok=True)

# Create wandb callback
wandb_callback = WandbCallback(
    model_save_freq=100000,  # Save every 100,000 steps
    model_save_path=model_dir,
    verbose=2
)

# Custom callback to save the best model
class SaveBestModelCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.best_model_path = os.path.join(model_dir, "best_model")

    def _on_step(self) -> bool:
        # Log episode rewards to check for the best model
        if "episode" in self.locals:
            episode_info = self.locals["infos"][0].get("episode", {})
            if "r" in episode_info:
                mean_reward = np.mean(episode_info["r"])
                wandb.log({"mean_reward": mean_reward}, step=self.num_timesteps)
                
                # Save the best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"New best mean reward: {mean_reward}. Saving model...")
                    self.model.save(self.best_model_path)
                    wandb.log({"best_mean_reward": mean_reward}, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Upload the best model to WandB
        print(f"Uploading the best model with mean reward: {self.best_mean_reward}")
        wandb.save(f"{self.best_model_path}.zip")

save_best_callback = SaveBestModelCallback()

# Total training timesteps per iteration
time_steps = 6000000

# Training loop
for i in range(10):
    print(f"Starting learn iteration {i + 1}")
    
    # Train the model and log data
    model.learn(
        total_timesteps=time_steps,
        callback=[wandb_callback, save_best_callback],
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
    print(f"Completed learn iteration {i + 1}")
    
    # Save the model after each iteration
    model.save(f"{model_dir}/{time_steps * (i + 1)}")
    print(f"Model saved at iteration {i + 1}: {model_dir}/{time_steps * (i + 1)}")