"""
MovingV2Aviary: Stage 3 of curriculum - sphere starts offset and moves in fixed direction.
Starting position: [0.3, 0.5, 0.7], moves right (+Y) at 6cm/s.
Extended observation: drone state (12) + target position (3) + target velocity (3) = 18
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gymnasium import spaces
import numpy as np
import pybullet as p


class MovingV2Aviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05
        self.TARGET_MASS = 0.5
        self.MOVE_SPEED = 0.06  # 6cm/s
        
        self.TARGET_START = np.array([0.3, 0.5, 0.7])
        self.MOVE_DIRECTION = np.array([0.3, 1.0, 0.0])
        self.MOVE_DIRECTION = self.MOVE_DIRECTION / np.linalg.norm(self.MOVE_DIRECTION)
        
        self.TARGET_POS = self.TARGET_START.copy()
        self.TARGET_VEL = self.MOVE_DIRECTION * self.MOVE_SPEED
        
        self.collision_occurred = False
        self._addTargetSphere()
        
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
            state[0:3],
            state[7:10],
            state[10:13],
            state[13:16],
        ])
        obs = np.hstack([base_obs, self.TARGET_POS, self.TARGET_VEL]).astype(np.float32)
        return obs

    def _addTargetSphere(self):
        """Create a sphere with mass."""
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            physicsClientId=self.CLIENT
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            rgbaColor=[0.2, 0.6, 1.0, 0.9],  # Blue
            physicsClientId=self.CLIENT
        )
        self.TARGET_ID = p.createMultiBody(
            baseMass=self.TARGET_MASS,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT
        )
        p.changeDynamics(
            self.TARGET_ID, -1,
            linearDamping=0.9,
            angularDamping=0.9,
            physicsClientId=self.CLIENT
        )

    def _moveTarget(self):
        """Move the target each step."""
        if not hasattr(self, 'TARGET_ID'):
            return
        time_elapsed = self.step_counter / self.PYB_FREQ
        new_pos = self.TARGET_START + self.MOVE_DIRECTION * self.MOVE_SPEED * time_elapsed
        new_pos[0] = np.clip(new_pos[0], -1.5, 1.5)
        new_pos[1] = np.clip(new_pos[1], -1.5, 1.5)
        new_pos[2] = np.clip(new_pos[2], 0.3, 1.5)
        self.TARGET_POS = new_pos
        
        p.resetBasePositionAndOrientation(
            self.TARGET_ID, new_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(
            self.TARGET_ID,
            linearVelocity=self.TARGET_VEL.tolist(),
            angularVelocity=[0, 0, 0],
            physicsClientId=self.CLIENT
        )

    def _computeReward(self):
        """Distance reward + collision bonus."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        self._moveTarget()
        
        norm = np.linalg.norm(self.TARGET_POS - pos)
        reward = max(0, 4 - norm**2)
        
        direction = (self.TARGET_POS - pos) / (norm + 1e-6)
        approach_vel = np.dot(vel, direction)
        reward += 0.5 * max(0, approach_vel)
        
        if norm < 0.2 and hasattr(self, 'TARGET_ID'):
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
        if abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 4 or state[2] < 0:
            return True
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {"collision_occurred": self.collision_occurred, "target_pos": self.TARGET_POS.tolist()}

    def reset(self, seed=None, options=None):
        self.collision_occurred = False
        self.TARGET_POS = self.TARGET_START.copy()
        result = super().reset(seed=seed, options=options)
        self._addTargetSphere()
        return result
