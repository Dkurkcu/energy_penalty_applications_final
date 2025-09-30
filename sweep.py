# sweep.py
import os
import yaml
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.wrappers import TimeLimit


def train():
    wandb.init()
    config = wandb.config

    # HIDDEN INDEX LOGIC (episode x-axis without showing the index chart)
    wandb.define_metric("ep/step", hidden=True)        # hidden x-axis for episode-level charts
    wandb.define_metric("ep/*", step_metric="ep/step") # any metric under ep/* uses ep/step on x-axis

    from reacher_v3 import ReacherV3Env  # Import inside for isolation

    # Unique log dir for each run
    log_dir = f"./sweep_logs/{wandb.run.id}"
    os.makedirs(log_dir, exist_ok=True)

    def make_env(seed=None):
        def _init():
            env = ReacherV3Env("reacher_v3.xml")

            # Proper seeding for Gymnasium + SB3
            if seed is not None:
                try:
                    env.action_space.seed(int(seed))
                except Exception:
                    pass
                try:
                    env.reset(seed=int(seed))
                except Exception:
                    pass

            env = TimeLimit(env, max_episode_steps=400)

            # Log only what we actually use
            env = Monitor(
                env,
                filename=os.path.join(log_dir, "monitor.csv"),
                info_keywords=(
                    "episode_energy",
                    "success",
                    "distance",
                    
                ),
            )
            return env
        return _init

    # ---------- Minimal logging: success rate, episode reward/energy ----------
    class ReacherLoggingCallback(BaseCallback):
        def __init__(self, window=100, verbose=0):
            super().__init__(verbose)
            self.window = int(window)
            self._cur_success = None
            self.ep_success = []
            # Added for episode-indexed charts
            self._step_in_ep = None
            self._first_success_step = None
            self._episode_index = 0

        def _on_training_start(self) -> None:
            n_envs = getattr(self.training_env, "num_envs", 1)
            self._cur_success = [False] * n_envs
            self._step_in_ep = [0] * n_envs
            self._first_success_step = [None] * n_envs

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for i, info in enumerate(infos):
                # track steps within current episode & first success step
                self._step_in_ep[i] += 1
                if info.get("success", False) and self._first_success_step[i] is None:
                    self._first_success_step[i] = self._step_in_ep[i]
                    self._cur_success[i] = True

                # episode end?
                if "episode" in info and "r" in info["episode"]:
                    ep_r = float(info["episode"]["r"])
                    ep_success = 1 if self._cur_success[i] else 0
                    self._cur_success[i] = False

                    # rolling success rate
                    self.ep_success.append(ep_success)
                    sr_window = float(np.mean(self.ep_success[-self.window:])) if self.ep_success else 0.0

                    # energy from env info
                    ep_energy = float(info.get("episode_energy", 0.0))

                    # original timestep-indexed log (unchanged)
                    wandb.log(
                        {
                            "success_rate": sr_window,
                            "episode_reward": ep_r,
                            "episode_energy": ep_energy,
                        },
                        step=self.num_timesteps,
                    )

                    # HIDDEN INDEX LOG: episode-indexed charts using hidden ep/step
                    final_distance = float(np.asarray(info.get("distance", np.nan)))
                    steps_to_success = (
                        float(self._first_success_step[i])
                        if self._first_success_step[i] is not None else float("nan")
                    )
                    self._episode_index += 1
                    wandb.log(
                        {
                            "ep/step": self._episode_index,         # hidden x-axis
                            "ep/final_distance": final_distance,    # y-values
                            "ep/steps_to_success": steps_to_success,
                            "ep/success": ep_success,
                        }
                    )

                    # reset per-episode trackers for env i
                    self._step_in_ep[i] = 0
                    self._first_success_step[i] = None

            return True

    # Seed from sweep config
    seed_value = int(config.seed)
    np.random.seed(seed_value)

    venv = DummyVecEnv([make_env(seed=seed_value)])
    try:
        venv.seed(seed_value)
    except Exception:
        pass

    # Hyperparams from sweep
    learning_rate = float(config.learning_rate)
    ent_coef = float(config.ent_coef)
    batch_size = int(config.batch_size)
    clip_range = float(config.clip_range)

    # Optional single env for manual checks (not used by the model)
    env = ReacherV3Env("reacher_v3.xml")
    try:
        env.action_space.seed(seed_value)
        env.reset(seed=seed_value)
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
        verbose=1,
    )

    # Eval callback (simple)
    eval_seed = seed_value + 10_000
    eval_env = DummyVecEnv([make_env(seed=eval_seed)])
    try:
        eval_env.seed(eval_seed)
    except Exception:
        pass

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=5,
        verbose=1,
    )

    model.learn(
        total_timesteps=4_000_000,
        callback=[ReacherLoggingCallback(window=100), eval_callback],
    )

    model.save("fixed_arm_directsweep.zip")
    wandb.finish()
    venv.close()


if __name__ == "__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project="energy_tracking_with_penalties")
    wandb.agent(sweep_id, function=train)
