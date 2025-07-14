import time
import numpy as np
from reacher_v3 import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

def make_env():
    env = ReacherV3Env("reacher_v3.xml", render_mode="human")
    env = TimeLimit(env, max_episode_steps=200)  # Longer episode for easier learning/viewing
    return env

venv = DummyVecEnv([make_env])
model = PPO.load("ppo_reacher_v3", env=venv)
obs = venv.reset()

for episode in range(100):  # Number of episodes to watch
    done = False
    truncated = False
    episode_reward = 0.0
    steps = 0
    last_info = {}

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        venv.render()
        time.sleep(0.01)

        # Unpack VecEnv results (always batch axis 0)
        done = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done
        reward = reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
        info = info[0] if isinstance(info, (list, tuple, np.ndarray)) else info

        episode_reward += reward
        steps += 1
        last_info = info

        # For Gymnasium TimeLimit truncation
        truncated = info.get('TimeLimit.truncated', False)

    # Robustly convert distance to float (handles array/scalar)
    dist = float(np.asarray(last_info.get('distance', float('nan'))))

    if last_info.get("success", False):
        print(f"EPISODE {episode+1:2d}: SUCCESS! Distance: {dist:.4f} | Reward: {episode_reward:.2f} | Steps: {steps}")
    elif truncated:
        print(f"EPISODE {episode+1:2d}: TRUNCATED.  Distance: {dist:.4f} | Reward: {episode_reward:.2f} | Steps: {steps}")
    else:
        print(f"EPISODE {episode+1:2d}: ??? (Unexpected End) Distance: {dist:.4f} | Reward: {episode_reward:.2f} | Steps: {steps}")

    # Reset for the next episode
    obs = venv.reset()
    time.sleep(0.5)
