import time
import numpy as np
from reacher_v3 import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

# Match the training energy settings here
USE_ENERGY_PENALTY = True
ENERGY_COEF = 0.1

def make_env():
    env = ReacherV3Env(
        "reacher_v3.xml",
        render_mode="human",
        use_energy_penalty=USE_ENERGY_PENALTY,
        energy_penalty_coef=ENERGY_COEF
    )
    env = TimeLimit(env, max_episode_steps=200)  # keep your viewing length
    return env

venv = DummyVecEnv([make_env])

# Optional sanity check: verify what the env actually uses
print("Energy penalty ON?:", venv.get_attr("use_energy_penalty")[0],
      "| coef:", venv.get_attr("energy_penalty_coef")[0])

model = PPO.load("fixed_arm_directsweep.zip", env=venv)
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
