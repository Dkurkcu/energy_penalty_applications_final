import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from reacher_v3 import ReacherV3Env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Load the sweep configuration from YAML
with open("sweep_config.yaml", 'r') as file:
    sweep_configuration = yaml.safe_load(file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_configuration, project="Reacher-Curriculum")

class ReacherLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []  # Track rewards per episode
        self.success_flags = []    # Track success (1 if success condition is met, else 0)
        self.window = 100          # Window size for moving average

    def _on_step(self) -> bool:
        # Access `info` passed into the callback
        infos = self.locals['infos']  # This contains the 'info' of all environments

        for info in infos:
            # Check if 'episode' exists in the info
            if 'episode' in info:
                # Extract reward and success status
                reward = info['episode']['r']  # This is the reward for the current episode
                # Safely access the success key
                success = info['episode'].get('success', False)  # Default to False if 'success' is missing

                # Track the success rate based on some condition (e.g., distance threshold)
                self.success_flags.append(1 if success else 0)

                # Log episode reward and success rate
                wandb.log({
                    "episode_reward": reward,
                    "success_rate": np.mean(self.success_flags[-self.window:]),  # Moving average of success rate
                }, step=self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        if len(self.episode_rewards) == 0:
            return

        rewards = self.episode_rewards
        episodes = list(range(len(rewards)))  # Create list of episodes

        # Log Episode Reward Over Time (raw values)
        wandb.log({
            "episode_reward_over_time": rewards,
        })

        # Log Moving Success Rate (raw values)
        moving_success_rate = np.convolve(self.success_flags, np.ones(self.window) / self.window, mode='valid')
        wandb.log({
            "moving_success_rate_plot": moving_success_rate,
        })

# Function for training during the sweep (without saving the model)
def train_sweep_model():
    # Initialize WandB at the start of the function
    wandb.init()  # Ensure WandB is initialized before using wandb.config

    # Load the pre-trained model (ppo_reacher_0.03.zip)
    model = PPO.load("donut_outer_0.05.zip", env=create_venv())  # Model trained at r=0.03

    # Apply the hyperparameters from the sweep for this trial
    model.learning_rate = wandb.config.learning_rate
    model.ent_coef = wandb.config.ent_coef
    model.batch_size = wandb.config.batch_size  # If batch_size is also part of the sweep

    # Train the model with the new hyperparameters from the sweep
    model.learn(total_timesteps=100_000, callback=[ReacherLoggingCallback()])  # Use custom callback to log performance

    # Finish WandB logging without saving the model
    wandb.finish()

# Function to create environment (no radius change here, will use the manual reset_model changes)
def make_env():
    env = ReacherV3Env("reacher_v3.xml")  # r=0.04 will be set manually in reset_model()
    env = Monitor(env, filename="monitor.csv")  # Wrap with Monitor for logging
    return env

# Create the vectorized environment (using the default radius in reset_model)
def create_venv():
    return DummyVecEnv([lambda: make_env()])

# Start the sweep using WandB agent
wandb.agent(sweep_id, function=train_sweep_model)