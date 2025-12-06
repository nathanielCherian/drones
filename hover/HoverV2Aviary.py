"""
HoverV2Aviary: Stage 1 of curriculum - hover up to stationary sphere at [0, 0, 1].
Extended observation: drone state (12) + target position (3) + target velocity (3) = 18
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gymnasium import spaces
import numpy as np
import pybullet as p


class HoverV2Aviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        self.TARGET_VEL = np.array([0.0, 0.0, 0.0])  # Stationary
        
        self.collision_occurred = False
        self._addTargetSphere()
        
        # Override observation space
        self.observation_space = self._observationSpace()

    def _observationSpace(self):
        """Extended observation: base (12) + target_pos (3) + target_vel (3) = 18."""
        low = np.array([-np.inf] * 18, dtype=np.float32)
        high = np.array([np.inf] * 18, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        """Compute observation: drone state + target position + target velocity."""
        state = self._getDroneStateVector(0)
        base_obs = np.hstack([
            state[0:3],    # position
            state[7:10],   # rpy
            state[10:13],  # velocity
            state[13:16],  # angular velocity
        ])
        obs = np.hstack([base_obs, self.TARGET_POS, self.TARGET_VEL]).astype(np.float32)
        return obs

    def _addTargetSphere(self):
        """Create a visible target sphere."""
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            physicsClientId=self.CLIENT
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            rgbaColor=[0.2, 0.8, 0.2, 0.9],  # Green
            physicsClientId=self.CLIENT
        )
        self.TARGET_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT
        )

    def _computeReward(self):
        """Distance reward + collision bonus."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        norm = np.linalg.norm(self.TARGET_POS - pos)
        reward = max(0, 4 - norm**2)
        
        # Approach velocity bonus
        direction = (self.TARGET_POS - pos) / (norm + 1e-6)
        approach_vel = np.dot(vel, direction)
        reward += 0.5 * max(0, approach_vel)
        
        # Collision check
        if norm < 0.15 and hasattr(self, 'TARGET_ID'):
            contacts = p.getContactPoints(
                bodyA=self.DRONE_IDS[0],
                bodyB=self.TARGET_ID,
                physicsClientId=self.CLIENT
            )
            if len(contacts) > 0:
                reward += 500.0
                self.collision_occurred = True

        return reward

    def _computeTerminated(self):
        return self.collision_occurred

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 4 or state[2] < 0:
            return True
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"collision_occurred": self.collision_occurred}

    def reset(self, seed=None, options=None):
        self.collision_occurred = False
        result = super().reset(seed=seed, options=options)
        self._addTargetSphere()
        return result
