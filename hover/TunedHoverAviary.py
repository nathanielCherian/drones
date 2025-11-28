from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np

class TunedHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.TARGET_POS = np.array([0, 0, 1])
        self.EPISODE_LEN_SEC = 8
        self.TARGET_QUAT = np.array([0, 0, 0, 1])  # xyzw

    def _computeReward(self):
        """Computes the current reward value.

        Combines penalties for distance and angle difference with bonuses for
        reaching the target position.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        
        # Calculate position error
        pos_error = self.TARGET_POS - state[0:3]
        vel_error = np.linalg.norm(state[10:13])
        distance = np.linalg.norm(pos_error)

        
        # Base reward: penalize distance and angle difference
        distance_penalty = distance ** 2  # Quadratic penalty for distance
        vel_penalty = vel_error ** 2       # Quadratic penalty for velocity

        #compute distance from hover for rpy stabilization
        rpy_error = state[9:12]
        rpy_penalty = np.linalg.norm(rpy_error) ** 2
        
        base_reward = -1.0 * distance_penalty - 1.0 * vel_penalty - 0.5 * rpy_penalty

        if self._computeTruncated():
            base_reward -= 10.0  # Large penalty for truncation (crash or out of bounds)
        else:
            base_reward += 0.001  # Small reward for surviving each step
        
        # # Bonus for being close to target position
        # if distance < 0.2:
        #     base_reward += 5.0  # Strong bonus when very close
        # elif distance < 0.4:
        #     base_reward += 2.0  # Moderate bonus when close
        
        # # Bonus for good orientation alignment
        # if angle_diff < 0.3:  # ~17 degrees
        #     base_reward += 1.0
        
        return float(base_reward)

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
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 4 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False