"""
ForceHitAviary: Target sphere at [0,0,1] (same as TunedHoverAviary).
Drone learns to hit the sphere with MAXIMUM FORCE.
Reward is based on collision normal force - hit it hard!
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import pybullet as p


class ForceHitAviary(HoverAviary):

    def __init__(self, *args, target_pos=None, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30
        self.TARGET_RADIUS = 0.05  # 5cm radius sphere
        
        # Target at same position as TunedHoverAviary [0, 0, 1]
        if target_pos is not None:
            self.TARGET_POS = np.array(target_pos)
        else:
            self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        
        self.collision_occurred = False
        self.collision_force = 0.0
        self.max_force_this_episode = 0.0
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
        """Distance reward + BIG force-based collision bonus."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        
        # Distance-based reward (same as TunedHoverAviary)
        norm = np.linalg.norm(self.TARGET_POS - pos)
        reward = max(0, 4 - norm**2)
        
        # Approach velocity bonus (encourage building speed toward target)
        direction = (self.TARGET_POS - pos) / (norm + 1e-6)
        approach_vel = np.dot(vel, direction)
        reward += 0.5 * max(0, approach_vel)  # Reward approaching fast
        
        # Only check collision if close enough
        if norm < 0.175 and hasattr(self, 'TARGET_ID'):
            contacts = p.getContactPoints(
                bodyA=self.DRONE_IDS[0],
                bodyB=self.TARGET_ID,
                physicsClientId=self.CLIENT
            )
            if len(contacts) > 0:
                # Print each contact point's force for debugging
                forces = [c[9] for c in contacts]
                print(f"  Contact points: {len(contacts)}, forces: {forces}")
                
                # Sum up normal forces from all contact points
                total_force = sum(forces)
                self.collision_force = total_force
                self.max_force_this_episode = max(self.max_force_this_episode, total_force)
                
                # Base collision bonus
                base_bonus = 500.0
                
                # Force multiplier: scale force to meaningful reward
                # Forces are tiny (0.001-0.1N), so multiply by 20000 to make it matter
                # A 0.01N hit = +200 bonus, 0.05N hit = +1000 bonus
                force_bonus = total_force * 20000.0
                
                reward += base_bonus + force_bonus
                self.collision_occurred = True
                print(f"[COLLISION] Force: {total_force:.4f}N -> bonus: {base_bonus + force_bonus:.1f}")
        
        return reward

    def _computeTerminated(self):
        """Terminate on collision with target sphere."""
        return self.collision_occurred

    def _computeTruncated(self):
        """Truncate if out of bounds, too tilted, or timeout."""
        state = self._getDroneStateVector(0)
        
        # Out of bounds
        if abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 4 or state[2] < 0:
            return True
        
        # Too tilted
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        
        # Timeout
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        return False

    def _computeInfo(self):
        """Return collision force info."""
        return {
            "collision_occurred": self.collision_occurred,
            "collision_force": self.collision_force,
            "max_force_this_episode": self.max_force_this_episode,
        }

    def reset(self, seed=None, options=None):
        """Reset and re-add the target sphere."""
        self.collision_occurred = False
        self.collision_force = 0.0
        self.max_force_this_episode = 0.0
        result = super().reset(seed=seed, options=options)
        self._addTargetSphere()
        return result
