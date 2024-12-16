import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from ot2_gym_wrapper import OT2Env
import subprocess
subprocess.run(["pip", "install", "--upgrade", "typing_extensions"], check=True)

# ClearML initialization
task = Task.init(project_name="Mentor Group K/Group 1/MichonGoddijn", task_name="RL_PPO_Experiment")
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")
task.set_requirements(["pydantic==2.10.3", "typing_extensions==4.9.0"])

# Weights & Biases initialization
import wandb

wandb.init(
    project="RL_OT2_Control",
    config={
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
    },
    sync_tensorboard=True,
)

# Environment setup
env = OT2Env(render=False, max_steps=1000)

# RL algorithm
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=wandb.config.learning_rate,
    n_steps=wandb.config.n_steps,
    batch_size=wandb.config.batch_size,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    ent_coef=wandb.config.ent_coef,
    clip_range=wandb.config.clip_range,
    max_grad_norm=wandb.config.max_grad_norm,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
)

# Callback for checkpoints and tracking
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./checkpoints/", name_prefix="rl_model")
wandb_callback = WandbCallback()

# Train the model
model.learn(total_timesteps=500000, callback=[checkpoint_callback, wandb_callback])

# Save the model
model.save("ppo_ot2_model")
wandb.save("ppo_ot2_model.zip")

# Close the environment
env.close()

# Finalize WandB
wandb.finish()
