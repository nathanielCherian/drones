from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import pybullet as p

class TunedHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05  # Radius of the target sphere

        self._addTargetSphere()

    def _addTargetSphere(self):
        """Create a large visible sphere as the collision target."""
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            physicsClientId=self.CLIENT
        )
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            rgbaColor=[1, 0.2, 0.2, 0.8],  # Red, slightly transparent
            physicsClientId=self.CLIENT
        )
        
        self.TARGET_ID = p.createMultiBody(
            baseMass=0,  # Static (won't move when hit)
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT
        )

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        # ret = -np.linalg.norm(self.TARGET_POS-state[0:3])**8
        norm = np.linalg.norm(self.TARGET_POS-state[0:3])
        ret = max(0, 4 - (norm)**2)
        #if np.linalg.norm(self.TARGET_POS-state[0:3]) < .4:
        #     ret *= 2
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
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 4 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def reset(self, seed=None, options=None):
        """Reset the environment and re-add the target sphere."""
        result = super().reset(seed=seed, options=options)
        self._addTargetSphere()
        return result