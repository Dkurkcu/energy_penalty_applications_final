import os
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle


class ReacherV3Env(MujocoEnv, EzPickle):
    """
    Reacher with:
      • Mechanical work tracking only: ΔE_k = tau_{k-1} · (q_k - q_{k-1})
      • Normalized observations in [-1, 1]
      • Reward with progress shaping (baseline ~ 0):
            progress = prev_dist - dist
            reward = progress -energy_cost- 0.0001 - 0.0001*||qdot|| - 0.001*||action||^2
        (plus success bonus: +10 if dist < 0.05, +5 extra if dist < 0.02, with a short delay)
      • No energy penalties/weights/warm-ups.
    """

    def __init__(self, xml_file="reacher_v3.xml", frame_skip=1, render_mode=None):
        EzPickle.__init__(self, xml_file, frame_skip, render_mode)
        fullpath = os.path.join(os.path.dirname(__file__), xml_file)

        # Bounds for min–max normalization (first 10 features)
        qpos_low  = np.array([-np.pi, -np.pi], dtype=np.float64)
        qpos_high = np.array([ np.pi,  np.pi], dtype=np.float64)
        qvel_low  = np.array([-3.0, -3.0], dtype=np.float64)
        qvel_high = np.array([ 3.0,  3.0], dtype=np.float64)
        r_tip = 1.2
        ft_low  = np.array([-r_tip, -r_tip, 0.0], dtype=np.float64)
        ft_high = np.array([ r_tip,  r_tip, 0.2], dtype=np.float64)
        tgt_low  = np.array([-0.9, -0.9, 0.0], dtype=np.float64)
        tgt_high = np.array([ 0.9,  0.9, 0.2], dtype=np.float64)

        self._feat_low_fixed  = np.concatenate([qpos_low, qvel_low, ft_low,  tgt_low])
        self._feat_high_fixed = np.concatenate([qpos_high, qvel_high, ft_high, tgt_high])
        self._E_min, self._E_max = -1.0, 1.0
        self._norm_eps = 1e-8

        MujocoEnv.__init__(
            self,
            model_path=fullpath,
            frame_skip=frame_skip,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32),
            render_mode=render_mode,
        )

        # Success delay machinery (for clean termination after success)
        self.success_delay_steps = 20
        self.current_delay = 0
        self.delay_pending_termination = False

        # Energy/work accounting (uses previous torque)
        self._prev_qpos = None
        self._prev_tau  = None
        self.episode_energy = 0.0

        # Progress shaping state
        self.prev_dist = None

        # For stable logging in delay window
        self._last_act_penalty = 0.0
        self._last_progress = 0.0

        # RNG
        self.np_random = np.random.RandomState()

    # ---------- helpers ----------
    def _minmax_to_minus1_1(self, x, lo, hi):
        s01 = (x - lo) / (np.maximum(hi - lo, self._norm_eps))
        return np.clip(2.0 * s01 - 1.0, -1.0, 1.0)

    def _build_raw_obs(self):
        qpos = self.data.qpos[:2].astype(np.float64)
        qvel = self.data.qvel[:2].astype(np.float64)
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id].astype(np.float64)
        target_pos = self.model.site("target").pos.astype(np.float64)
        epE = np.array([self.episode_energy], dtype=np.float64)
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos, epE])

    def _get_normalized_obs(self):
        raw = self._build_raw_obs()
        norm_fixed = self._minmax_to_minus1_1(raw[:10], self._feat_low_fixed, self._feat_high_fixed)
        E = float(raw[10])
        # dynamic min–max for the energy channel (observability only)
        if np.isfinite(E):
            if E < self._E_min: self._E_min = E
            if E > self._E_max: self._E_max = E
        if abs(self._E_max - self._E_min) < 1e-6:
            self._E_max = self._E_min + 1e-6
        E_norm = self._minmax_to_minus1_1(np.array([E]), np.array([self._E_min]), np.array([self._E_max]))[0]
        return np.concatenate([norm_fixed, np.array([E_norm], dtype=np.float64)]).astype(np.float32)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(int(seed))
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def seed(self, seed_value=None):
        self.np_random = np.random.RandomState(seed_value)
        return [seed_value]

    def get_obs(self):
        return self._get_normalized_obs()

    def _get_obs(self):
        return self.get_obs()

    def reset_model(self):
        # Start pose
        fixed_qpos = np.array([0.0, 0.0], dtype=np.float64)
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        self.set_state(fixed_qpos, qvel)

        # Random target on donut (r ∈ [0.2, 0.9])
        inner_radius = 0.2
        outer_radius = 0.9
        rng = getattr(self, "np_random", np.random)
        r2 = rng.uniform(inner_radius**2, outer_radius**2)
        r = np.sqrt(r2)
        theta = rng.uniform(0, 2*np.pi)
        offset = r * np.array([np.cos(theta), np.sin(theta)])
        target_pos = np.array([offset[0], offset[1], 0.1])
        self.model.site_pos[self.model.site("target").id][:] = target_pos

        self.do_simulation(np.zeros(self.model.nu), 1)

        # Reset trackers
        self.episode_energy = 0.0
        self._E_min, self._E_max = -1.0, 1.0

        self._prev_qpos = self.data.qpos[:2].copy()
        self._prev_tau  = np.zeros(2, dtype=np.float64)

        # initialize prev_dist for progress shaping
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        self.prev_dist = float(np.linalg.norm(fingertip_pos - target_pos))

        # reset delay window state
        self.current_delay = 0
        self.delay_pending_termination = False

        # reset cached logging values
        self._last_act_penalty = 0.0
        self._last_progress = 0.0

        return self.get_obs()

    def step(self, action):
        # success delay window
        if self.current_delay > 0:
            self.current_delay -= 1
            fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
            target_pos = self.model.site("target").pos
            dist = float(np.linalg.norm(fingertip_pos - target_pos))

            reward = 0.0
            terminated = False
            truncated = False
            info = {
                "delay": True,
                "distance": dist,
                "success": dist < 0.05,
                "episode_energy": self.episode_energy,
                "act_penalty": float(self._last_act_penalty),
                "progress": float(self._last_progress),
            }
            if self.current_delay == 0 and self.delay_pending_termination:
                terminated = True
                self.delay_pending_termination = False
            return self.get_obs(), reward, terminated, truncated, info

        # Step simulation
        self.do_simulation(action, self.frame_skip)

        # Mechanical work increment: tau_{k-1} · (q_k - q_{k-1})
        qpos_now = self.data.qpos[:2].copy()
        dq = qpos_now - (self._prev_qpos if self._prev_qpos is not None else qpos_now)
        tau_prev = self._prev_tau if self._prev_tau is not None else np.zeros(2, dtype=np.float64)
        energy_inc = float(np.dot(tau_prev, dq))
        self.episode_energy += energy_inc

        # Prepare next history tau_k
        self._prev_qpos = qpos_now
        self._prev_tau  = self.data.qfrc_actuator[:2].copy()

        # Read state
        qvel = self.data.qvel[:2]
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        dist = float(np.linalg.norm(fingertip_pos - target_pos))

        # ===================== reward (progress shaping; literals here) =====================
        progress     = float(self.prev_dist - dist)     # >0 if getting closer, <0 if farther
        time_penalty = 0.0001
        vel_penalty  = 0.0001 * float(np.linalg.norm(qvel))
        a = np.asarray(action, dtype=np.float64)
        act_penalty  = 0.001 * float(np.dot(a, a))      
        energy_cost  = 0.05 * abs(energy_inc)                 # <-- energy weight (tune 0.01–0.2)

        reward = progress - energy_cost - time_penalty - vel_penalty - act_penalty
        # success reward + start delay window for clean termination
        success = dist < 0.05
        if success:
            reward += 10.0
            if dist < 0.02:
                reward += 5.0
            self.current_delay = self.success_delay_steps
            self.delay_pending_termination = True
        # =====================================================================

        # update progress state and cached logging values
        self.prev_dist = dist
        self._last_act_penalty = act_penalty
        self._last_progress = progress

        info = {
            "distance": dist,
            "success": success,
            "progress": progress,
            "energy_inc": energy_inc,
            "episode_energy": self.episode_energy,
            "act_penalty": act_penalty,
        }

        terminated = False
        truncated = False
        return self.get_obs(), reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        return super().close()
