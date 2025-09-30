import time
import numpy as np
from reacher_v3 import ReacherV3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit


def make_env():
    def _init():
        env = ReacherV3Env("reacher_v3.xml", render_mode="human")
        # shorter viewing length so you can watch more episodes quickly
        env = TimeLimit(env, max_episode_steps=200)
        return env
    return _init


# Single-env VecEnv with human render
venv = DummyVecEnv([make_env()])

# Load your trained policy
model = PPO.load("fixed_arm_directsweep.zip", env=venv)

# Reset
obs = venv.reset()

for episode in range(100):  # number of episodes to watch
    done = False
    truncated = False
    ep_reward = 0.0
    steps = 0
    last_info = {}

    while not (done or truncated):
        # policy step (deterministic for viewing)
        action, _ = model.predict(obs, deterministic=True)

        # env step
        obs, reward, done_arr, infos = venv.step(action)

        # render the mujoco viewer
        venv.render()
        time.sleep(0.01)

        # unwrap VecEnv outputs (batch size = 1)
        done = bool(done_arr[0]) if isinstance(done_arr, (list, tuple, np.ndarray)) else bool(done_arr)
        reward = float(reward[0]) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
        info = infos[0] if isinstance(infos, (list, tuple, np.ndarray)) else infos

        ep_reward += reward
        steps += 1
        last_info = info

        # Gymnasium TimeLimit flag
        truncated = bool(info.get("TimeLimit.truncated", False))

    # summarize episode
    dist = float(np.asarray(last_info.get("distance", np.nan)))
    success = bool(last_info.get("success", False))

    tag = "SUCCESS!" if success else ("TRUNCATED." if truncated else "DONE.")
    print(f"EPISODE {episode+1:02d}: {tag}  Distance(last)={dist:.4f}  Return={ep_reward:.2f}  Steps={steps}")

    # reset for next episode
    obs = venv.reset()
    time.sleep(0.4)
