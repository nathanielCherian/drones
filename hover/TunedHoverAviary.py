from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym.spaces import Box
import numpy as np

class TunedHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 10
        #self.TARGET_POS = np.array([0, 0, 1])
        
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)
        # Keep TARGET_POS as a 1D array [x, y, z] to match usage elsewhere
        self.TARGET_POS = np.array([x, y, z])
        self.TARGET_RPY = np.array([0, 0, 0])
        # Store initial xyzs and rpys ranges for randomization on reset
        self._randomize_init_pos = True
        self._randomize_target_pos = True
        # If the parent set an observation space, expand it to include the
        # target position (3 values). We append the target to the end of the
        # state vector so existing positional indices remain valid.
        try:
            if hasattr(self, "observation_space") and isinstance(self.observation_space, Box):
                old_shape = self.observation_space.shape
                if len(old_shape) == 1:
                    new_shape = (old_shape[0] + 3,)
                elif len(old_shape) == 2:
                    # (n_agents, state_dim) -> add 3 to state_dim
                    new_shape = (old_shape[0], old_shape[1] + 3)
                else:
                    new_shape = old_shape
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=new_shape, dtype=np.float32)
        except Exception:
            # If anything goes wrong (package differences), skip silently.
            pass

    def reset(self, **kwargs):
        """Reset the environment and randomize initial position if enabled."""
        if self._randomize_init_pos:
            # Randomize x, y in [-1, 1] and z in [0, 2] for each reset
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0, 2)
            self.INIT_XYZS = np.array([[x, y, z]])
            # Keep rotations at zero
            self.INIT_RPYS = np.array([[0, 0, 0]])
        else:
            self.INIT_RPYS = np.array([[0, 0, 0]])
            self.INIT_XYZS = np.array([[0, 0, 1.15]])
        if self._randomize_target_pos:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0, 1)
            # Keep TARGET_POS as a 1D array [x, y, z] to match usage elsewhere
            self.TARGET_POS = np.array([x, y, z])
        print(self.TARGET_POS)
        # Call parent reset with any kwargs (e.g., seed)
        ret = super().reset(**kwargs)

        # Parent may return (obs, info) (gym>=0.26) or just obs
        obs = None
        info = None
        if isinstance(ret, tuple):
            if len(ret) >= 1:
                obs = ret[0]
            if len(ret) >= 2:
                info = ret[1]
        else:
            obs = ret

        # Append target position to the observation if it's a numpy array
        try:
            tp = np.asarray(self.TARGET_POS).reshape(3,)
        except Exception:
            tp = np.array([0.0, 0.0, 0.0])

        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                obs = np.concatenate((obs, tp))
            elif obs.ndim == 2:
                # Assume shape (n_agents, state_dim) or (batch, state_dim)
                tp_row = tp.reshape(1, 3)
                tp_tile = np.tile(tp_row, (obs.shape[0], 1))
                obs = np.hstack((obs, tp_tile))

        # Return the same structure the parent returned
        if info is None:
            return obs
        else:
            return (obs, info)

    def _computeReward(self):
        """Computes the current reward value.

        Combines penalties for distance and angle difference with bonuses for
        reaching the target position.

        Returns
        -------
        float
            The reward.

        """
        # Acquire state and velocity
        state = self._getDroneStateVector(0)
        # Note: in this environment velocity components are at indices 10:13
        vel = state[10:13]
        pos = state[0:3]

        # Distance to target
        norm = np.linalg.norm(self.TARGET_POS - pos)

        # Base distance reward (linear with distance)
        distance_reward = max(0.0, 4.0 - norm)

        # Vertical offset relative to target (positive means above target)
        dz = pos[2] - self.TARGET_POS[2]

        # Reward motion toward the target altitude (non-quadratic):
        # term = -k * dz * vz  => positive when velocity is directed toward target
        vz = float(vel[2])
        vel_towards_reward = -2.0 * dz * vz
        vel_towards_reward = float(np.clip(vel_towards_reward, -3.0, 3.0))

        # Penalize overall speed to encourage smooth hovering
        speed = float(np.linalg.norm(vel))
        speed_penalty = 0.2 * speed

        # Bonus for being very close and nearly stationary
        close_bonus = 0.0
        if norm < 0.2 and speed < 0.2:
            close_bonus = 1.5

        # Small stabilizing bonus when near target (encourages low velocity)
        near_vel_bonus = 0.0
        if norm < 0.2:
            near_vel_bonus = max(0.0, 1.0 - speed) * 2.0

        # Combine terms
        ret = distance_reward + vel_towards_reward + close_bonus + near_vel_bonus - speed_penalty

        return float(ret)

    def _getDroneStateVector(self, drone_id: int):
        """Return the drone state vector with the target position appended.

        This keeps the original ordering (so existing index-based code still
        works) and appends three values corresponding to the target XYZ.
        """
        state = super()._getDroneStateVector(drone_id)
        try:
            tp = np.asarray(self.TARGET_POS).reshape(3,)
        except Exception:
            tp = np.array([0.0, 0.0, 0.0])
        return np.concatenate((state, tp))

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        return False
        # state = self._getDroneStateVector(0)
        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
        #     return True
        # else:
        #     return False


    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 2 or abs(state[1]) > 2 or state[2] > 5 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False