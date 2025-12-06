"""
CardinalV2Aviary: Stage 4 of curriculum - randomized cardinal direction movement.
Sphere moves in one of 4 directions (right, left, forward, back), randomized each episode.
Starting position is offset in direction of movement.
Extended observation: drone state (12) + target position (3) + target velocity (3) = 18
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gymnasium import spaces
import numpy as np
import pybullet as p


class CardinalV2Aviary(HoverAviary):
    
    DIRECTIONS = {
        "right":   np.array([0.0,  1.0, 0.0]),
        "left":    np.array([0.0, -1.0, 0.0]),
        "forward": np.array([1.0,  0.0, 0.0]),
        "back":    np.array([-1.0, 0.0, 0.0]),
    }
    
    STARTING_OFFSETS = {
        "right":   np.array([0.0,  0.4, 0.7]),
        "left":    np.array([0.0, -0.4, 0.7]),
        "forward": np.array([0.4,  0.0, 0.7]),
        "back":    np.array([-0.4, 0.0, 0.7]),
    }

    def __init__(self, *args, fixed_direction=None, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05
        self.TARGET_MASS = 0.5
        self.MOVE_SPEED = 0.06
        
        self.fixed_direction = fixed_direction
        self.current_direction_name = None
        self.MOVE_DIRECTION = None
        self.TARGET_START = None
        self.TARGET_POS = None
        self.TARGET_VEL = np.zeros(3)
        
        self.collision_occurred = False
        
        self._randomize_direction()
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
        target_pos = self.TARGET_POS if self.TARGET_POS is not None else np.zeros(3)
        target_vel = self.TARGET_VEL if self.TARGET_VEL is not None else np.zeros(3)
        obs = np.hstack([base_obs, target_pos, target_vel]).astype(np.float32)
        return obs

    def _randomize_direction(self):
        """Pick a random cardinal direction."""
        if self.fixed_direction is not None:
            direction_name = self.fixed_direction
        else:
            direction_name = np.random.choice(list(self.DIRECTIONS.keys()))
        
        self.current_direction_name = direction_name
        self.MOVE_DIRECTION = self.DIRECTIONS[direction_name].copy()
        self.TARGET_START = self.STARTING_OFFSETS[direction_name].copy()
        self.TARGET_POS = self.TARGET_START.copy()
        self.TARGET_VEL = self.MOVE_DIRECTION * self.MOVE_SPEED

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
            rgbaColor=[1.0, 0.5, 0.0, 0.9],  # Orange
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
        """Move the target in its cardinal direction."""
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
        return {
            "collision_occurred": self.collision_occurred,
            "target_pos": self.TARGET_POS.tolist(),
            "direction": self.current_direction_name,
        }

    def reset(self, seed=None, options=None):
        self.collision_occurred = False
        result = super().reset(seed=seed, options=options)
        self._randomize_direction()
        self._addTargetSphere()
        return result
