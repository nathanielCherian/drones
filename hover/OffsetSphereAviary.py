"""
OffsetSphereAviary: Target sphere is offset from directly above.
Drone must learn to tilt and move to hit the sphere.
Based on TunedHoverAviary but with offset target and collision termination.
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import pybullet as p


class OffsetSphereAviary(HoverAviary):

    def __init__(self, *args, target_pos=None, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05  # 5cm radius sphere
        
        # Offset target: forward, right, and a bit lower than [0,0,1]
        if target_pos is not None:
            self.TARGET_POS = np.array(target_pos)
        else:
            self.TARGET_POS = np.array([0.3, 0.3, 0.7])  # Default offset position
        
        self.collision_occurred = False
        self._addTargetSphere()

    def _addTargetSphere(self):
        """Create a visible sphere as the collision target."""
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
        """Same reward as TunedHoverAviary + collision bonus."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # Distance-based reward (same as TunedHoverAviary)
        norm = np.linalg.norm(self.TARGET_POS - pos)
        reward = max(0, 4 - norm**2)
        
        # Only check collision if close enough (cheap distance check first)
        # Drone radius ~0.05m + target radius 0.05m + small margin = 0.15m
        if norm < 0.175 and hasattr(self, 'TARGET_ID'):
            contacts = p.getContactPoints(
                bodyA=self.DRONE_IDS[0],
                bodyB=self.TARGET_ID,
                physicsClientId=self.CLIENT
            )
            if len(contacts) > 0:
                reward += 500.0  # Big bonus for hitting the target!
                self.collision_occurred = True
                print(f"[COLLISION] Hit target! Distance was {norm:.4f}m")
        
        return reward

    def _computeTerminated(self):
        """Terminate on collision with target sphere (success!)."""
        # Reuse the flag set in _computeReward() to avoid duplicate collision check
        return self.collision_occurred

    def _computeTruncated(self):
        """Truncate if out of bounds, too tilted, or timeout."""
        state = self._getDroneStateVector(0)
        
        # Out of bounds
        if abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 4:
            return True
        
        # Too tilted (but allow more tilt than TunedHoverAviary since we need to move)
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        
        # Timeout
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False

    def reset(self, seed=None, options=None):
        """Reset and re-add the target sphere."""
        self.collision_occurred = False
        result = super().reset(seed=seed, options=options)
        self._addTargetSphere()
        return result
