from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np

class TunedHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 10
        self.TARGET_POS = np.array([0, 0, 1])
        self.TARGET_RPY = np.array([0, 0, 0])
        # Store initial xyzs and rpys ranges for randomization on reset
        self._randomize_init_pos = True

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
        
        # Call parent reset with any kwargs (e.g., seed)
        return super().reset(**kwargs)

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