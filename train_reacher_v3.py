import os
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from reacher_v3 import ReacherV3Env

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# --- Energy penalty control
USE_ENERGY_PENALTY = True
ENERGY_COEF = 0.1  # Adjust this to penalize high energy usage

def make_env():
    # Pass energy config into the env PPO trains on
    env = ReacherV3Env(
        "reacher_v3.xml",
        use_energy_penalty=USE_ENERGY_PENALTY,
        energy_penalty_coef=ENERGY_COEF
    )
    env.energy_coef = ENERGY_COEF  # keep your original line; mapped to energy_penalty_coef
    env = TimeLimit(env, max_episode_steps=300)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"),
                  info_keywords=("episode_energy", "success"))
    return env

venv = DummyVecEnv([make_env])
check_env(venv.envs[0], warn=True)

wandb.init(
    project="Reacher-Energy_plotting",
    name="seed_0_energyeff_penaltytrue",
    config={
        "env_name": "ReacherV3",
        "total_timesteps": 1_000_000,
        "learning_rate": 2e-5,
        "ent_coef": 0.002,
        "batch_size": 256,
        "n_epochs": 15,
        "clip_range": 0.02,
        "gamma": 0.95,
        "gae_lambda": 0.85,
        "energy_coef": ENERGY_COEF,
    },
)

config = wandb.config

# ✅ Logging callback (OLD metric: reward-threshold success)
class ReacherLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.success_flags = []
        self.window = 100
        self.reward_success_threshold = 7.5  # unchanged

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info and "r" in info["episode"]:
                reward = info["episode"]["r"]

                self.episode_rewards.append(reward)
                self.success_flags.append(1 if reward >= self.reward_success_threshold else 0)

                log_dict = {
                    "episode_reward": reward,
                    "success_rate": np.mean(self.success_flags[-self.window:]),
                }
                if "episode_energy" in info:
                    try:
                        log_dict["episode_energy"] = float(info["episode_energy"])
                    except Exception:
                        pass

                wandb.log(log_dict, step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if len(self.episode_rewards) == 0:
            return
        wandb.log({
            "episode_reward_over_time": self.episode_rewards,
            "moving_success_rate_plot": np.convolve(
                self.success_flags, np.ones(self.window) / self.window, mode='valid'
            ),
        })

# ✅ Energy annealing callback (kept)
class EnergyAnneal(BaseCallback):
    def __init__(self, start=0.02, end=1.0, ramp_steps=120_000, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.ramp_steps = start, end, ramp_steps

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_energy_coef", float(self.start))
        wandb.log({"energy_coef": float(self.start)}, step=self.num_timesteps)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / float(self.ramp_steps))
        coef = float(self.start + frac * (self.end - self.start))
        self.training_env.env_method("set_energy_coef", coef)
        wandb.log({"energy_coef": coef}, step=self.num_timesteps)
        return True

# Set reproducibility 
seed_value = 0
np.random.seed(seed_value)

# Create environment for training (single env for checks/logging)
env = ReacherV3Env(
    "reacher_v3.xml",
    use_energy_penalty=USE_ENERGY_PENALTY,
    energy_penalty_coef=ENERGY_COEF
)
env.energy_coef = ENERGY_COEF  # keep your original line; mapped to energy_penalty_coef
env = Monitor(env, info_keywords=("episode_energy", "success"))

policy_kwargs = dict(net_arch=[256, 256, 256])

# Load previous model and continue training with new energy penalty
model = PPO.load(
    "bothrandom9try.zip",
    env=venv,  # make sure new env with energy penalty is used
    learning_rate=config.learning_rate,
    ent_coef=config.ent_coef,
    batch_size=config.batch_size,
    clip_range=config.clip_range,
    n_epochs=config.n_epochs,
    gae_lambda=config.gae_lambda,
    seed=seed_value,
    verbose=1,
)

# Train with annealing + old logger
model.learn(
    total_timesteps=config.total_timesteps,
    callback=[ReacherLoggingCallback(), EnergyAnneal(start=0.02, end=ENERGY_COEF, ramp_steps=120_000)]
)

# Save new model
model.save("bothrandom10_energy.zip")
wandb.finish()
venv.close()
