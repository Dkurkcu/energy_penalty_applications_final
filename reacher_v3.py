import numpy as np
import os
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

class ReacherV3Env(MujocoEnv, EzPickle):
    """
    Adds mechanical energy tracking:
      ΔE_k = sum_j tau_{j,k-1} * (q_{j,k} - q_{j,k-1})
    and an optional per-step penalty:  -lambda * |ΔE_k|
    """
    def __init__(
        self,
        xml_file="reacher_v3.xml",
        frame_skip=1,
        render_mode=None,
        use_energy_penalty=True,       # toggle energy objective
        energy_penalty_coef=1e-3       # penalty strength (tune!)
    ):
        EzPickle.__init__(self, xml_file, frame_skip, render_mode, use_energy_penalty, energy_penalty_coef)
        fullpath = os.path.join(os.path.dirname(__file__), xml_file)
        MujocoEnv.__init__(
            self,
            model_path=fullpath,
            frame_skip=frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            render_mode=render_mode,
        )
        # Your existing flags
        self.success_delay_steps = 20
        self.current_delay = 0
        self.delay_pending_termination = False

        # Energy objective config
        self.use_energy_penalty = bool(use_energy_penalty)
        self.energy_penalty_coef = float(energy_penalty_coef)

        # Energy accounting state
        self._prev_qpos = None         # q_{k-1} (first 2 joints)
        self._prev_tau  = None         # tau_{k-1} (first 2 joints)
        self.episode_energy = 0.0

    # --- Alias to support training script's env.energy_coef usage
    @property
    def energy_coef(self):
        return self.energy_penalty_coef

    @energy_coef.setter
    def energy_coef(self, v):
        self.energy_penalty_coef = float(v)

    # --- Setter for annealing (used by EnergyAnneal callback)
    def set_energy_coef(self, v: float):
        self.energy_penalty_coef = float(v)

    # Keep your legacy seed method (unchanged)
    def seed(self, seed_value=None):
    # Use the seed only for RNG, never for physics values
        self.np_random, seed_value = np.random.RandomState(seed_value), seed_value
        return [seed_value]


    def get_obs(self):
        qpos = self.data.qpos[:2]
        qvel = self.data.qvel[:2]
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos]).astype(np.float32)

    def _get_obs(self):
        return self.get_obs()

    def reset_model(self):
        fixed_qpos = np.array([0.0, 0.0], dtype=np.float64)  # e.g., shoulder=0.0, elbow=0.0
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        self.set_state(fixed_qpos, qvel)
        #random_qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=2)
        #qvel = np.zeros(self.model.nv)
        #self.set_state(random_qpos, qvel)

        inner_radius = 0.2
        outer_radius = 0.9
        r2 = np.random.uniform(inner_radius**2, outer_radius**2)
        r = np.sqrt(r2)
        theta = np.random.uniform(0, 2*np.pi)
        offset = r * np.array([np.cos(theta), np.sin(theta)])
        target_pos = np.array([offset[0], offset[1], 0.1])
        self.model.site_pos[self.model.site("target").id][:] = target_pos

        self.do_simulation(np.zeros(self.model.nu), 1)

        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        self.prev_dist = np.linalg.norm(fingertip_pos - target_pos)

        self.current_delay = 0
        self.delay_pending_termination = False

        # Reset energy accounting
        self._prev_qpos = self.data.qpos[:2].copy()
        self._prev_tau  = np.zeros(2, dtype=np.float64)   # tau_{k-1} = 0 at first step
        self.episode_energy = 0.0

        return self.get_obs()

    def step(self, action):
        # Delay window branch (your logic)
        if self.current_delay > 0:
            self.current_delay -= 1
            obs = self.get_obs()
            fingertip_pos = obs[4:7]
            target_pos = obs[7:10]
            dist = np.linalg.norm(fingertip_pos - target_pos)
            reward = 0.0
            terminated = False
            truncated = False
            info = {
                "delay": True,
                "distance": dist,
                "success": dist < 0.05,
                "episode_energy": self.episode_energy,
            }
            if self.current_delay == 0 and self.delay_pending_termination:
                terminated = True
                self.delay_pending_termination = False
            return obs, reward, terminated, truncated, info

        # Simulate
        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()

        # ---- Mechanical energy increment: tau_{k-1} · (q_k - q_{k-1})
        qpos_now = self.data.qpos[:2].copy()
        dq = qpos_now - (self._prev_qpos if self._prev_qpos is not None else qpos_now)
        tau_prev = self._prev_tau if self._prev_tau is not None else np.zeros(2, dtype=np.float64)
        energy_inc = float(np.dot(tau_prev, dq))
        self.episode_energy += energy_inc

        # Prepare next history tau_k from current actuator generalized forces
        self._prev_qpos = qpos_now
        self._prev_tau  = self.data.qfrc_actuator[:2].copy()  # or actuator_force[:2]

        # ---- Your reward terms
        qvel = self.data.qvel[:2]
        fingertip_pos = obs[4:7]
        target_pos = obs[7:10]
        dist = np.linalg.norm(fingertip_pos - target_pos)
        distance_reward = getattr(self, "prev_dist", dist) - dist
        velocity_penalty = 0.001 * np.linalg.norm(qvel)
        action_penalty   = 0.00005 * np.linalg.norm(action)  # reduced a bit (energy term now handles effort)
        time_penalty     = 0.001

        reward = distance_reward - velocity_penalty - action_penalty - time_penalty

        # ---- Energy penalty (toggleable)
        if self.use_energy_penalty:
            reward -= self.energy_penalty_coef * abs(energy_inc)

        success = dist < 0.05
        terminated = False
        truncated = False
        info = {"distance": dist, "success": success}

        if success:
            reward += 10.0
            if dist < 0.02:
                reward += 5.0
            self.current_delay = self.success_delay_steps
            self.delay_pending_termination = True
            info["delay"] = True

        # Always expose energy
        info["energy_inc"] = energy_inc
        info["episode_energy"] = self.episode_energy

        self.prev_dist = dist
        return obs, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        return super().close()
