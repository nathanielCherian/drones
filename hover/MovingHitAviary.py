"""
MovingHitAviary: A sphere with MASS that moves slowly.
- Sphere has mass, so PyBullet computes real collision forces
- Sphere moves very slowly in a direction
- Drone learns to hit it hard for maximum force reward
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import pybullet as p


class MovingHitAviary(HoverAviary):

    def __init__(self, *args, target_pos=None, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05  # 5cm radius sphere
        self.TARGET_MASS = 0.5    # Give it some mass (500g) for real physics
        
        # Starting position - same as OffsetSphereAviary but slightly more to the right
        if target_pos is not None:
            self.TARGET_START = np.array(target_pos)
        else:
            self.TARGET_START = np.array([0.3, 0.5, 0.7])  # Shifted right (+Y)
        
        self.TARGET_POS = self.TARGET_START.copy()
        
        # Movement parameters - 6cm/s, moving slightly to the right
        self.MOVE_SPEED = 0.06  # 6cm per second
        self.MOVE_DIRECTION = np.array([0.3, 1.0, 0.0])  # Mostly right (+Y), slight forward (+X)
        self.MOVE_DIRECTION = self.MOVE_DIRECTION / np.linalg.norm(self.MOVE_DIRECTION)  # Normalize
        
        self.collision_occurred = False
        self.collision_force = 0.0
        self.max_force_this_episode = 0.0
        self._addTargetSphere()

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
            rgbaColor=[0.2, 1.0, 0.2, 0.9],  # Green to distinguish from ForceHit
            physicsClientId=self.CLIENT
        )
        
        self.TARGET_ID = p.createMultiBody(
            baseMass=self.TARGET_MASS,  # HAS MASS! Real physics!
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT
        )
        
        # Disable gravity for the sphere (so it floats but still has mass for collisions)
        p.changeDynamics(
            self.TARGET_ID,
            -1,  # -1 for base link
            linearDamping=0.9,   # High damping so it doesn't fly away too fast when hit
            angularDamping=0.9,
            physicsClientId=self.CLIENT
        )

    def _moveTarget(self):
        """Move the target slowly each step."""
        if not hasattr(self, 'TARGET_ID'):
            return
            
        # Calculate new position based on time
        time_elapsed = self.step_counter / self.PYB_FREQ
        new_pos = self.TARGET_START + self.MOVE_DIRECTION * self.MOVE_SPEED * time_elapsed
        
        # Keep it within bounds
        new_pos[0] = np.clip(new_pos[0], -1.0, 1.0)
        new_pos[1] = np.clip(new_pos[1], -1.0, 1.0)
        new_pos[2] = np.clip(new_pos[2], 0.3, 1.5)
        
        self.TARGET_POS = new_pos
        
        # Set position and also give it the velocity (for more realistic physics)
        p.resetBasePositionAndOrientation(
            self.TARGET_ID,
            new_pos.tolist(),
            [0, 0, 0, 1],  # No rotation
            physicsClientId=self.CLIENT
        )
        
        # Set velocity to match movement
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
        
        # Move the target first
        self._moveTarget()
        
        # Distance-based reward
        norm = np.linalg.norm(self.TARGET_POS - pos)
        reward = max(0, 4 - norm**2)
        
        # Approach velocity bonus
        direction = (self.TARGET_POS - pos) / (norm + 1e-6)
        approach_vel = np.dot(vel, direction)
        reward += 0.5 * max(0, approach_vel)
        
        # Collision check - only if close
        if norm < 0.2 and hasattr(self, 'TARGET_ID'):
            contacts = p.getContactPoints(
                bodyA=self.DRONE_IDS[0],
                bodyB=self.TARGET_ID,
                physicsClientId=self.CLIENT
            )
            if len(contacts) > 0:
                # With mass, we should get real forces!
                forces = [c[9] for c in contacts]
                total_force = sum(forces)
                
                self.collision_force = total_force
                self.max_force_this_episode = max(self.max_force_this_episode, total_force)
                
                # Base collision bonus
                base_bonus = 1000.0
                
                # Force bonus - with real mass, forces should be more meaningful
                # Typical drone collision might be 0.1-5N range
                force_bonus = total_force * 100.0  # 1N = +100 reward
                
                reward += base_bonus + force_bonus
                self.collision_occurred = True
                print(f"[MOVING HIT] Force: {total_force:.4f}N, Total bonus: {base_bonus + force_bonus:.1f}")
        
        return reward

    def _computeTerminated(self):
        """Terminate on collision."""
        return self.collision_occurred

    def _computeTruncated(self):
        """Truncate if out of bounds, too tilted, or timeout."""
        state = self._getDroneStateVector(0)
        
        # Out of bounds (wider range since target moves)
        if abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 4 or state[2] < 0:
            return True
        
        # Too tilted
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        
        # Timeout
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False

    def _computeInfo(self):
        """Return collision and target info."""
        return {
            "collision_occurred": self.collision_occurred,
            "collision_force": self.collision_force,
            "max_force_this_episode": self.max_force_this_episode,
            "target_pos": self.TARGET_POS.tolist(),
        }

    def reset(self, seed=None, options=None):
        """Reset and re-add the target sphere."""
        self.collision_occurred = False
        self.collision_force = 0.0
        self.max_force_this_episode = 0.0
        self.TARGET_POS = self.TARGET_START.copy()
        result = super().reset(seed=seed, options=options)
        self._addTargetSphere()
        return result
