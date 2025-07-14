import numpy as np
import os
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

class ReacherV3Env(MujocoEnv, EzPickle):
    def __init__(self, xml_file="reacher_v3.xml", frame_skip=1, render_mode=None):
        EzPickle.__init__(self, xml_file, frame_skip, render_mode)
        fullpath = os.path.join(os.path.dirname(__file__), xml_file)
        MujocoEnv.__init__(
            self,
            model_path=fullpath,
            frame_skip=frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
            render_mode=render_mode,
        )
        self.success_delay_steps = 20
        self.current_delay = 0
        self.delay_pending_termination = False

    def get_obs(self):
        qpos = self.data.qpos[:2]
        qvel = self.data.qvel[:2]
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        target_pos = self.model.site("target").pos
        return np.concatenate([qpos, qvel, fingertip_pos, target_pos]).astype(np.float32)

    def _get_obs(self):
        return self.get_obs()

    def reset_model(self):
        # Randomize starting position for the arm each time (small range)
        qpos = np.random.uniform(low=-1.0, high=1.0, size=2)
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        # --- FIXED TARGET POSITION: Always at (0.7, 0, 0.1) (well within reach) ---
        target_pos = np.array([0.7, 0.0, 0.1])
        target_site_id = self.model.site("target").id
        self.model.site_pos[target_site_id][:] = target_pos
        # Uniformly sample a target inside a circle of reach radius R=0.7
#R = 0.7
 #       theta = np.random.uniform(0, 2 * np.pi)
  #      r = R * np.sqrt(np.random.uniform(0, 1))
   #     x = r * np.cos(theta)
    #    y = r * np.sin(theta)
     #   target_pos = np.array([x, y, 0.1])
      #  target_site_id = self.model.site("target").id
       # self.model.site_pos[target_site_id][:] = target_pos

        # For reward shaping, use fingertip site
        fingertip_pos = self.data.site_xpos[self.model.site("fingertip_site").id]
        self.prev_dist = np.linalg.norm(fingertip_pos - target_pos)

        # Reset delay
        self.current_delay = 0
        self.delay_pending_termination = False

        return self.get_obs()

    def step(self, action):
        # Success delay logic
        if self.current_delay > 0:
            self.current_delay -= 1
            obs = self.get_obs()
            fingertip_pos = obs[4:7]
            target_pos = obs[7:10]
            dist = np.linalg.norm(fingertip_pos - target_pos)
            reward = 0.0
            terminated = False
            truncated = False
            info = {"delay": True, "distance": dist, "success": dist < 0.03}
            if self.current_delay == 0 and self.delay_pending_termination:
                terminated = True
                self.delay_pending_termination = False
            return obs, reward, terminated, truncated, info

        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()
        qvel = self.data.qvel[:2]
        fingertip_pos = obs[4:7]
        target_pos = obs[7:10]
        dist = np.linalg.norm(fingertip_pos - target_pos)
        distance_reward = getattr(self, "prev_dist", dist) - dist
        velocity_penalty = 0.005 * np.linalg.norm(qvel)
        action_penalty = 0.002 * np.linalg.norm(action)
        time_penalty = 0.01  # No time penalty

        reward = distance_reward - velocity_penalty - action_penalty - time_penalty

        success = dist < 0.03
        terminated = False
        truncated = False
        info = {"distance": dist, "success": success}

        if success:
            reward += 10.0
            self.current_delay = self.success_delay_steps
            self.delay_pending_termination = True
            info["delay"] = True

        self.prev_dist = dist

        return obs, reward, terminated, truncated, info

    def render(self):
        return super().render()

    def close(self):
        return super().close()
