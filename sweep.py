import os
import yaml
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.wrappers import TimeLimit

USE_ENERGY_PENALTY = True
ENERGY_COEF = 0.1

def train():
    wandb.init()
    config = wandb.config

    from reacher_v3 import ReacherV3Env  # Import here for total isolation

    # Unique log dir for each run
    log_dir = f"./sweep_logs/{wandb.run.id}"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(seed=None):
        def _init():
            env = ReacherV3Env(
                "reacher_v3.xml",
                use_energy_penalty=USE_ENERGY_PENALTY,
                energy_penalty_coef=ENERGY_COEF
            )
            env.energy_coef = ENERGY_COEF

            if seed is not None:
                try:
                    env.seed(int(seed))
                except Exception:
                    pass

            env = TimeLimit(env, max_episode_steps=400)
            env = Monitor(
                env,
                filename=os.path.join(log_dir, "monitor.csv"),
                info_keywords=("episode_energy", "success")
            )
            return env
        return _init

    class ReacherLoggingCallback(BaseCallback):
        def __init__(self, window=100, spike_threshold=9.5, verbose=0):
            super().__init__(verbose)
            self.window = int(window)
            self.spike_threshold = float(spike_threshold)
            self.episode_rewards = []
            self.success_flags = []  # your old reward-threshold SR (kept)
            self.spike_flags = []    # 1 if reward >= spike_threshold

            self.reward_success_threshold = 7.5  # unchanged

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info and "r" in info["episode"]:
                    r = float(info["episode"]["r"])
                    self.episode_rewards.append(r)
                    self.success_flags.append(1 if r >= self.reward_success_threshold else 0)
                    self.spike_flags.append(1 if r >= self.spike_threshold else 0)

                    # Compute rolling stats
                    w = self.window
                    # classic success_rate (kept for dashboards)
                    sr = float(np.mean(self.success_flags[-w:])) if len(self.success_flags) >= 1 else 0.0

                    # Improvement score: change in spike rate from first window to last window
                    if len(self.spike_flags) >= 2 * w:
                        first_rate = float(np.mean(self.spike_flags[:w]))
                        last_rate = float(np.mean(self.spike_flags[-w:]))
                        improvement_score = last_rate - first_rate
                    else:
                        first_rate = float("nan")
                        last_rate = float(np.mean(self.spike_flags[-w:])) if len(self.spike_flags) >= 1 else float("nan")
                        improvement_score = float("nan")  # not enough history yet

                    log_dict = {
                        "episode_reward": r,
                        "success_rate": sr,                 # old metric, still visible
                        "spike_rate_last_window": last_rate,
                        "spike_rate_first_window": first_rate,
                        "improvement_score": improvement_score,  # <-- SWEEP TARGET
                    }

                    # Log episode energy if present
                    if "episode_energy" in info:
                        try:
                            log_dict["episode_energy"] = float(info["episode_energy"])
                        except Exception:
                            pass

                    wandb.log(log_dict, step=self.num_timesteps)
            return True

    # Seed handling from sweep config
    seed_value = int(config.seed)
    np.random.seed(seed_value)

    venv = DummyVecEnv([make_env(seed=seed_value)])
    try:
        venv.seed(seed_value)
    except Exception:
        pass

    learning_rate = float(config.learning_rate)
    ent_coef = float(config.ent_coef)
    batch_size = int(config.batch_size)
    clip_range = float(config.clip_range)

    env = ReacherV3Env(
        "reacher_v3.xml",
        use_energy_penalty=USE_ENERGY_PENALTY,
        energy_penalty_coef=ENERGY_COEF
    )
    env.energy_coef = ENERGY_COEF
    try:
        env.seed(seed_value)
    except Exception:
        pass

    policy_kwargs = dict(net_arch=[256, 256, 256])

    model = PPO(
        "MlpPolicy",
        env=venv,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        batch_size=batch_size,
        clip_range=clip_range,
        seed=seed_value,
        verbose=1
    )

   
    eval_seed = seed_value + 10_000
    eval_env = DummyVecEnv([make_env(seed=eval_seed)])
    try:
        eval_env.seed(eval_seed)
    except Exception:
        pass

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=5,
        verbose=1
    )

    
    model.learn(
        total_timesteps=4_000_000,
        callback=[ReacherLoggingCallback(), eval_callback]
    )
    model.save("fixed_arm_directsweep.zip")
    wandb.finish()
    venv.close()


if __name__ == "__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project="fixedarmdirectsweepwith5seeds_penalty_on")
    wandb.agent(sweep_id, function=train)
