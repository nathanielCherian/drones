"""
TrueRandomAviary: Fully randomized target position and velocity.
- Random starting position within configurable bounds
- Random velocity direction (XY plane only)
- Velocity magnitude configurable (for curriculum)
- Physics-based sphere movement with constant Z height

Extended observation: drone state (12) + target position (3) + target velocity (3) = 18
"""

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gymnasium import spaces
import numpy as np
import pybullet as p


class TrueRandomAviary(HoverAviary):

    POS_Z_MIN = 0.4

    def __init__(self, *args,
                 min_speed=0.02,
                 max_speed=0.04,
                 max_xy=0.5,
                 max_z=0.8,
                 fixed_speed=None,
                 fixed_direction=None,
                 **kwargs):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.fixed_speed = fixed_speed
        self.fixed_direction = fixed_direction
        self.max_xy = max_xy
        self.max_z = max_z
        self.TARGET_RADIUS = 0.05
        self.TARGET_MASS = 0.5
        self.TARGET_ID = None
        self.TARGET_START = None
        self.TARGET_POS = None
        self.TARGET_VEL = None
        self.MOVE_DIRECTION = None
        self.current_speed = None
        self.collision_occurred = False
        self.top_speed = 0.0
        self.target_z_height = None
        
        super().__init__(**kwargs)
        
        self.EPISODE_LEN_SEC = 30
        self._randomize_target()
        self._addTargetSphere()
        self.observation_space = self._observationSpace()

    def _observationSpace(self):
        low = np.array([-np.inf] * 18, dtype=np.float32)
        high = np.array([np.inf] * 18, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        vel = state[10:13]
        speed = np.linalg.norm(vel)
        if speed > self.top_speed:
            self.top_speed = speed
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

    def _randomize_target(self):
        x = np.random.uniform(-self.max_xy, self.max_xy)
        y = np.random.uniform(-self.max_xy, self.max_xy)
        z = np.random.uniform(self.POS_Z_MIN, self.max_z)
        self.TARGET_START = np.array([x, y, z])
        self.TARGET_POS = self.TARGET_START.copy()
        self.target_z_height = z

        if self.fixed_direction is not None:
            self.MOVE_DIRECTION = np.array(self.fixed_direction)
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            self.MOVE_DIRECTION = np.array([np.cos(angle), np.sin(angle), 0.0])

        if self.fixed_speed is not None:
            self.current_speed = self.fixed_speed
        else:
            self.current_speed = np.random.uniform(self.min_speed, self.max_speed)

        self.TARGET_VEL = self.MOVE_DIRECTION * self.current_speed

    def _addTargetSphere(self):
        if self.TARGET_ID is not None:
            p.removeBody(self.TARGET_ID, physicsClientId=self.CLIENT)
            self.TARGET_ID = None

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            physicsClientId=self.CLIENT
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.TARGET_RADIUS,
            rgbaColor=[1.0, 0.2, 1.0, 0.9],
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
            linearDamping=0.0,
            angularDamping=0.0,
            physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(
            self.TARGET_ID,
            linearVelocity=self.TARGET_VEL.tolist(),
            angularVelocity=[0, 0, 0],
            physicsClientId=self.CLIENT
        )

    def _updateTargetState(self):
        if self.TARGET_ID is None:
            return
        pos, orn = p.getBasePositionAndOrientation(self.TARGET_ID, physicsClientId=self.CLIENT)
        new_pos = [pos[0], pos[1], self.target_z_height]
        p.resetBasePositionAndOrientation(
            self.TARGET_ID,
            new_pos,
            orn,
            physicsClientId=self.CLIENT
        )
        p.resetBaseVelocity(
            self.TARGET_ID,
            linearVelocity=self.TARGET_VEL.tolist(),
            angularVelocity=[0, 0, 0],
            physicsClientId=self.CLIENT
        )
        self.TARGET_POS = np.array(new_pos)

   # def _computeReward(self):
   #     state = self._getDroneStateVector(0)
   ##     pos = state[0:3]
    #    vel = state[10:13]

    #    self._updateTargetState()

     #   norm = np.linalg.norm(self.TARGET_POS - pos)
      #  reward = max(0, 4 - norm**2)

     #   direction = (self.TARGET_POS - pos) / (norm + 1e-6)
     #   approach_vel = np.dot(vel, direction)
     #   reward += 0.5 * max(0, approach_vel)

     #   if norm < 0.2 and self.TARGET_ID is not None:
     #       contacts = p.getContactPoints(
     #           bodyA=self.DRONE_IDS[0],
     #           bodyB=self.TARGET_ID,
     #           physicsClientId=self.CLIENT
     #       )
     #       if len(contacts) > 0:
     #           reward += 500.0
     #           self.collision_occurred = True

    #    return reward
    #def _computeReward(self):
    #    state = self._getDroneStateVector(0)
    ##    pos = state[0:3]
     #   vel = state[10:13]

     #   self._updateTargetState()

        # Vector to target
    #    to_target = self.TARGET_POS - pos
    #    norm = np.linalg.norm(to_target)
    #    direction = to_target / (norm + 1e-6)
        
        # Base distance reward
    #    reward = max(0, 4 - norm**2)
        
        # Approach velocity reward
    #    approach_vel = np.dot(vel, direction)
    #    reward += 0.5 * max(0, approach_vel)
        
        # === NEW: Discourage being below target ===
    #    z_diff = pos[2] - self.TARGET_POS[2]  # positive if drone is above
    #    if z_diff < -0.1:  # drone is more than 10cm below target
    #        reward -= 0.5 * abs(z_diff)  # penalty scales with how far below
        
        # === NEW: Reward horizontal speed toward target ===
    #    horizontal_to_target = to_target.copy()
    #    horizontal_to_target[2] = 0
    #    horizontal_dist = np.linalg.norm(horizontal_to_target)
    #    if horizontal_dist > 0.01:
    #        horizontal_dir = horizontal_to_target / horizontal_dist
    #        horizontal_vel = np.dot(vel[:2], horizontal_dir[:2])
    #        reward += 0.3 * max(0, horizontal_vel)
        
        # === NEW: Reward being at similar height (ready to ram) ===
    #    if abs(z_diff) < 0.15:  # within 15cm of target height
    #        reward += 0.5
        
        # Collision bonus
    #    if norm < 0.2 and self.TARGET_ID is not None:
    #        contacts = p.getContactPoints(
    #            bodyA=self.DRONE_IDS[0],
    #            bodyB=self.TARGET_ID,
    #            physicsClientId=self.CLIENT
    #        )
    #        if len(contacts) > 0:
    #            reward += 500.0
    #            self.collision_occurred = True

     #   return reward



    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        self._updateTargetState()

        to_target = self.TARGET_POS - pos
        norm = np.linalg.norm(to_target)
        direction = to_target / (norm + 1e-6)
        
        # Estimate time to intercept (rough)
        drone_speed = np.linalg.norm(vel) + 0.1
        time_to_intercept = norm / drone_speed
        
        # Predict where target will be
        predicted_target_pos = self.TARGET_POS + self.TARGET_VEL * time_to_intercept
        
        # Vector to predicted position
        to_predicted = predicted_target_pos - pos
        predicted_norm = np.linalg.norm(to_predicted)
        predicted_dir = to_predicted / (predicted_norm + 1e-6)
        
        # Base reward: distance to current target
        reward = max(0, 4 - norm**2)
        
        # Reward velocity toward PREDICTED position (intercept)
        intercept_approach = np.dot(vel, predicted_dir)
        reward += 0.8 * max(0, intercept_approach)
        
        # Penalize being below target
        z_diff = pos[2] - self.TARGET_POS[2]
        if z_diff < -0.1:
            reward -= 0.5 * abs(z_diff)
        
        # Bonus for being at target height
        if abs(z_diff) < 0.15:
            reward += 0.5
        
        # Collision bonus
        if norm < 0.2 and self.TARGET_ID is not None:
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
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {
            "collision_occurred": self.collision_occurred,
            "target_pos": self.TARGET_POS.tolist() if self.TARGET_POS is not None else [0, 0, 0],
            "target_vel": self.TARGET_VEL.tolist() if self.TARGET_VEL is not None else [0, 0, 0],
            "speed": self.current_speed,
            "top_drone_speed": self.top_speed,
        }

    def set_speed_range(self, min_speed, max_speed, max_xy=None, max_z=None):
        self.min_speed = min_speed
        self.max_speed = max_speed
        if max_xy is not None:
            self.max_xy = max_xy
        if max_z is not None:
            self.max_z = min(max_z, 3.0)

    def reset(self, seed=None, options=None):
        self.collision_occurred = False
        self.top_speed = 0.0
        self.TARGET_ID = None
        result = super().reset(seed=seed, options=options)
        self._randomize_target()
        self._addTargetSphere()
        return result
