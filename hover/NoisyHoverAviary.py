from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import math

class NoisyHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        self.noise = kwargs['noise']
        del kwargs['noise']
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 15

    def step(self, action):
        noise = np.random.normal(loc=0, scale=self.noise, size=4)
        action = action + noise
        return super().step(action)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        pos_err = self.TARGET_POS-state[0:3]
        # pos_err[2] *= 5
        # euc_norm = np.linalg.norm(pos_err)
        # k = 2
        # ret = 10 * math.exp(-1 * k * (euc_norm**2))
        # ret = -np.linalg.norm(self.TARGET_POS-state[0:3])**8
        norm = np.linalg.norm(pos_err)
        ret = max(0, 4 - (norm)**2)
        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < .5:
        #     ret += 2
        return ret

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
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 1.5 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False