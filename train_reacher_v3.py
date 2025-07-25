from stable_baselines3 import PPO
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from reacher_v3 import ReacherV3Env
import os

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Create environment function
def make_env():
    env = ReacherV3Env("reacher_v3.xml")
    env = TimeLimit(env, max_episode_steps=200)  # Ensure episode ends
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))  #  wrap after TimeLimit
    return env

# Create the vectorized environment
venv = DummyVecEnv([make_env])

# Check if environment is valid
check_env(venv.envs[0], warn=True)

# Initialize Weights & Biases
wandb.init(
    project="Reacher-Donut-Shaped",
    name="0.02_---.100k_3e-4,0.01",
    config={
        "env_name": "ReacherV3",
        "total_timesteps": 100_000,
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
    },
)

# Define custom callback to log rewards and success
class ReacherLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []  # Track rewards per episode
        self.success_flags = []    # Track success (1 if reward > 7.5 else 0)
        self.window = 100          # Window size for moving average

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and "r" in info["episode"]:
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)
                self.success_flags.append(1 if reward >= 7.5 else 0)

                # Log episode reward
                wandb.log({
                    "episode_reward": reward,
                    "success_rate": np.mean(self.success_flags[-self.window:]),
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
# Set the random seed for reproducibility
seed_value = 42

# Set random seed for NumPy
np.random.seed(seed_value)

# Set random seed for the gym environment
env = ReacherV3Env("reacher_v3.xml")
env.seed(seed_value)
# Initialize PPO model
model = PPO("MlpPolicy", venv, seed=seed_value, verbose=1, learning_rate=3e-4, ent_coef=0.01 )

model = PPO.load("donut_outer_0.05.zip", env=venv)
# Train the model
model.learn(
    total_timesteps=100_000,
    callback=[ReacherLoggingCallback()]  # Only use our custom logging callback
)

# Save model and finish logging
#model.save("donut_0.02-0.06.zip")
wandb.finish()

# Close the environment
venv.close()
