"""
CardinalHitAviary: Sphere moves in one of 4 cardinal directions (left, right, forward, back).
- Direction is randomized each episode
- Starting position is offset in the direction of movement
- e.g., if moving right, starts on the right side
- EXTENDED OBSERVATION: includes target position and velocity so drone can track it!

This tests the drone's ability to intercept from any angle.
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gymnasium import spaces
import numpy as np
import pybullet as p


class CardinalHitAviary(HoverAviary):
    
    # The 4 cardinal directions: right, left, forward, back
    DIRECTIONS = {
        "right":   np.array([0.0,  1.0, 0.0]),   # +Y
        "left":    np.array([0.0, -1.0, 0.0]),   # -Y
        "forward": np.array([1.0,  0.0, 0.0]),   # +X
        "back":    np.array([-1.0, 0.0, 0.0]),   # -X
    }
    
    # Starting offsets for each direction (offset in direction of movement)
    STARTING_OFFSETS = {
        "right":   np.array([0.0,  0.4, 0.7]),   # Starts right of center
        "left":    np.array([0.0, -0.4, 0.7]),   # Starts left of center
        "forward": np.array([0.4,  0.0, 0.7]),   # Starts forward of center
        "back":    np.array([-0.4, 0.0, 0.7]),   # Starts back of center
    }

    def __init__(self, *args, fixed_direction=None, **kwargs):
        """
        Args:
            fixed_direction: If provided, use this direction instead of random.
                            Options: "right", "left", "forward", "back"
        """
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05  # 5cm radius sphere
        self.TARGET_MASS = 0.5    # Mass for real force computation
        self.MOVE_SPEED = 0.06    # 6cm per second
        
        self.fixed_direction = fixed_direction  # For eval with specific direction
        
        # Will be set in _randomize_direction()
        self.current_direction_name = None
        self.MOVE_DIRECTION = None
        self.TARGET_START = None
        self.TARGET_POS = None
        self.TARGET_VEL = np.zeros(3)  # Target velocity
        
        self.collision_occurred = False
        self.collision_force = 0.0
        self.max_force_this_episode = 0.0
        
        self._randomize_direction()
        self._addTargetSphere()
        
        # Override observation space to include target info
        self.observation_space = self._observationSpace()

    def _observationSpace(self):
        """Extended observation space: base (12) + target_pos (3) + target_vel (3) = 18."""
        # Base observation is 12: [pos(3), rpy(3), vel(3), ang_vel(3)]
        # We add: target_pos(3), target_vel(3) = 6 more
        low = np.array([-np.inf] * 18, dtype=np.float32)
        high = np.array([np.inf] * 18, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        """Compute observation: drone state + target position + target velocity."""
        state = self._getDroneStateVector(0)
        
        # Base observation (same as KIN): pos, rpy, vel, ang_vel
        base_obs = np.hstack([
            state[0:3],    # position
            state[7:10],   # rpy (roll, pitch, yaw)
            state[10:13],  # velocity
            state[13:16],  # angular velocity
        ])
        
        # Target info - position and velocity
        target_pos = self.TARGET_POS if self.TARGET_POS is not None else np.zeros(3)
        target_vel = self.TARGET_VEL if self.TARGET_VEL is not None else np.zeros(3)
        
        # Concatenate: [base_obs(12), target_pos(3), target_vel(3)] = 18
        obs = np.hstack([base_obs, target_pos, target_vel]).astype(np.float32)
        
        return obs

    def _randomize_direction(self):
        """Pick a random cardinal direction (or use fixed if specified)."""
        if self.fixed_direction is not None:
            direction_name = self.fixed_direction
        else:
            direction_name = np.random.choice(list(self.DIRECTIONS.keys()))
        
        self.current_direction_name = direction_name
        self.MOVE_DIRECTION = self.DIRECTIONS[direction_name].copy()
        self.TARGET_START = self.STARTING_OFFSETS[direction_name].copy()
        self.TARGET_POS = self.TARGET_START.copy()
        self.TARGET_VEL = self.MOVE_DIRECTION * self.MOVE_SPEED  # Velocity = direction * speed

    def _addTargetSphere(self):
        """Create a sphere WITH MASS so forces are computed."""
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            physicsClientId=self.CLIENT
        )
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            rgbaColor=[1.0, 0.5, 0.0, 0.9],  # Orange to distinguish
            physicsClientId=self.CLIENT
        )
        
        self.TARGET_ID = p.createMultiBody(
            baseMass=self.TARGET_MASS,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT
        )
        
        # Disable gravity, high damping
        p.changeDynamics(
            self.TARGET_ID,
            -1,
            linearDamping=0.9,
            angularDamping=0.9,
            physicsClientId=self.CLIENT
        )

    def _moveTarget(self):
        """Move the target in its cardinal direction each step."""
        if not hasattr(self, 'TARGET_ID'):
            return
            
        time_elapsed = self.step_counter / self.PYB_FREQ
        new_pos = self.TARGET_START + self.MOVE_DIRECTION * self.MOVE_SPEED * time_elapsed
        
        # Keep it within bounds
        new_pos[0] = np.clip(new_pos[0], -1.5, 1.5)
        new_pos[1] = np.clip(new_pos[1], -1.5, 1.5)
        new_pos[2] = np.clip(new_pos[2], 0.3, 1.5)
        
        self.TARGET_POS = new_pos
        
        p.resetBasePositionAndOrientation(
            self.TARGET_ID,
            new_pos.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )
        
        p.resetBaseVelocity(
            self.TARGET_ID,
            linearVelocity=(self.MOVE_DIRECTION * self.MOVE_SPEED).tolist(),
            angularVelocity=[0, 0, 0],
            physicsClientId=self.CLIENT
        )

    def _computeReward(self):
        """Distance reward + force-based collision bonus."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        self._moveTarget()
        
        # Distance-based reward
        norm = np.linalg.norm(self.TARGET_POS - pos)
        reward = max(0, 4 - norm**2)
        
        # Approach velocity bonus
        direction = (self.TARGET_POS - pos) / (norm + 1e-6)
        approach_vel = np.dot(vel, direction)
        reward += 0.5 * max(0, approach_vel)
        
        # Collision check
        if norm < 0.2 and hasattr(self, 'TARGET_ID'):
            contacts = p.getContactPoints(
                bodyA=self.DRONE_IDS[0],
                bodyB=self.TARGET_ID,
                physicsClientId=self.CLIENT
            )
            if len(contacts) > 0:
                forces = [c[9] for c in contacts]
                total_force = sum(forces)
                
                self.collision_force = total_force
                self.max_force_this_episode = max(self.max_force_this_episode, total_force)
                
                base_bonus = 200.0
                force_bonus = total_force * 100.0
                
                reward += base_bonus + force_bonus
                self.collision_occurred = True
                print(f"[CARDINAL HIT - {self.current_direction_name.upper()}] "
                      f"Force: {total_force:.4f}N, Bonus: {base_bonus + force_bonus:.1f}")
        
        return reward

    def _computeTerminated(self):
        """Terminate on collision."""
        return self.collision_occurred

    def _computeTruncated(self):
        """Truncate if out of bounds, too tilted, or timeout."""
        state = self._getDroneStateVector(0)
        
        if abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 4 or state[2] < 0:
            return True
        
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False

    def _computeInfo(self):
        """Return collision and direction info."""
        return {
            "collision_occurred": self.collision_occurred,
            "collision_force": self.collision_force,
            "max_force_this_episode": self.max_force_this_episode,
            "target_pos": self.TARGET_POS.tolist(),
            "direction": self.current_direction_name,
        }

    def reset(self, seed=None, options=None):
        """Reset, pick new random direction, and re-add sphere."""
        self.collision_occurred = False
        self.collision_force = 0.0
        self.max_force_this_episode = 0.0
        result = super().reset(seed=seed, options=options)
        self._randomize_direction()
        self._addTargetSphere()
        return result
