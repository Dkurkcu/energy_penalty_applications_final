from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from reacher_v3 import ReacherV3Env
import os

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    env = ReacherV3Env("reacher_v3.xml")
    env = TimeLimit(env, max_episode_steps=200)     # Ensure episode ends
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # ✅ wrap after TimeLimit
    return env

venv = DummyVecEnv([make_env])

check_env(venv.envs[0], warn=True)

model = PPO("MlpPolicy", venv, verbose=1)
model.learn(total_timesteps=500_000)

model.save("ppo_reacher_v3")
venv.close()
