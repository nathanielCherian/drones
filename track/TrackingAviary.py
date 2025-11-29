from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np
import pybullet as p

class TrackingAviary(HoverAviary):
    """
    A hovering environment with a moving target.
    The drone must reach a target location. Once it gets close enough,
    the target relocates to a random position at a fixed distance away.
    This repeats for a total of NUM_TARGET_CHANGES target relocations.
    """

    def __init__(self, num_target_changes=7, target_change_distance=1.0, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 30  # Total episode time
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        self.TARGET_RPY = np.array([0, 0, 0])
        
        # Tracking-specific parameters
        self.num_target_changes = num_target_changes  # How many times target relocates
        self.target_change_distance = target_change_distance  # Distance for next target
        self.target_changes_completed = 0  # Track how many times we've moved the target
        self.target_reached_threshold = 0.2  # Distance threshold to consider target reached
        
        # Store initial xyzs and rpys ranges for randomization on reset
        self._randomize_init_pos = True
        # Debug marker ids for target visualization in PyBullet
        self._target_marker_ids = []

    def reset(self, **kwargs):
        """Reset the environment and randomize initial position if enabled."""
        if self._randomize_init_pos:
            # Randomize x, y in [-1, 1] and z in [0, 2] for each reset
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0, 2)
            self.INIT_XYZS = np.array([[x, y, z]])
            # Keep rotations at zero
            self.INIT_RPYS = np.array([[0, 0, 0]])
        
        # Reset tracking state
        self.target_changes_completed = 0
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        # Create marker for initial target
        try:
            self._create_target_marker()
        except Exception:
            pass
        
        # Call parent reset with any kwargs (e.g., seed)
        return super().reset(**kwargs)

    def _computeReward(self):
        """Computes the current reward value for tracking task."""
        state = self._getDroneStateVector(0)
        vel = state[10:13]
        pos = state[0:3]

        # Distance to current target
        norm = np.linalg.norm(self.TARGET_POS - pos)

        # Base distance reward (linear with distance)
        distance_reward = max(0.0, 4.0 - norm)

        # Vertical offset relative to target (positive means above target)
        dz = pos[2] - self.TARGET_POS[2]

        # Reward motion toward the target altitude (non-quadratic):
        vz = float(vel[2])
        vel_towards_reward = -2.0 * dz * vz
        vel_towards_reward = float(np.clip(vel_towards_reward, -3.0, 3.0))

        # Penalize overall speed to encourage smooth movement
        speed = float(np.linalg.norm(vel))
        speed_penalty = 0.2 * speed

        # Bonus for being very close and nearly stationary (scales with closeness)
        close_bonus = 0.0
        if norm < 0.2 and speed < 0.2:
            # Bonus scales linearly: max 1.5 at target, decays to 0 at 0.2m away
            close_bonus = 1.5 * max(0.0, 1.0 - (norm / 0.2))

        # Small stabilizing bonus when near target (encourages low velocity)
        near_vel_bonus = 0.0
        if norm < 0.2:
            near_vel_bonus = max(0.0, 1.0 - speed) * 2.0

        # Combine terms
        ret = distance_reward + vel_towards_reward + close_bonus + near_vel_bonus - speed_penalty

        return float(ret)

    def _updateTarget(self):
        """Update target position if drone has reached current target."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # Check if drone reached current target
        dist_to_target = np.linalg.norm(self.TARGET_POS - pos)
        
        if dist_to_target < self.target_reached_threshold:
            # Only change target if we haven't exceeded the number of changes
            if self.target_changes_completed < self.num_target_changes:
                # Generate new target at random angle, fixed distance from current target
                angle = np.random.uniform(0, 2 * np.pi)
                elevation = np.random.uniform(-np.pi / 4, np.pi / 4)  # Restrict vertical angle
                
                dx = self.target_change_distance * np.cos(elevation) * np.cos(angle)
                dy = self.target_change_distance * np.cos(elevation) * np.sin(angle)
                dz = self.target_change_distance * np.sin(elevation)
                
                # New target position
                new_target = self.TARGET_POS + np.array([dx, dy, dz])
                
                # Clamp z to valid range [0.5, 3.5] to avoid going too low or too high
                new_target[2] = np.clip(new_target[2], 0.5, 3.5)
                
                self.TARGET_POS = new_target
                self.target_changes_completed += 1
                # Update visualization for new target
                try:
                    self._create_target_marker()
                except Exception:
                    pass

    def step(self, action):
        """Step the environment and update target."""
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update target position if needed
        self._updateTarget()
        
        return obs, reward, terminated, truncated, info

    def _create_target_marker(self):
        """Create or update a small crosshair marker at the current TARGET_POS in PyBullet GUI."""
        # Remove previous markers
        cid = self.getPyBulletClient()
        for idd in getattr(self, '_target_marker_ids', []):
            try:
                p.removeUserDebugItem(idd, physicsClientId=cid)
            except Exception:
                try:
                    p.removeUserDebugItem(idd)
                except Exception:
                    pass
        self._target_marker_ids = []

        # Draw three short lines centered at TARGET_POS
        center = self.TARGET_POS.tolist()
        size = 0.15
        # X axis line
        id1 = p.addUserDebugLine([center[0]-size, center[1], center[2]], [center[0]+size, center[1], center[2]], [1,0,0], 2, physicsClientId=cid)
        # Y axis line
        id2 = p.addUserDebugLine([center[0], center[1]-size, center[2]], [center[0], center[1]+size, center[2]], [0,1,0], 2, physicsClientId=cid)
        # Z axis line
        id3 = p.addUserDebugLine([center[0], center[1], center[2]-size], [center[0], center[1], center[2]+size], [0,0,1], 2, physicsClientId=cid)

        # Optional label
        try:
            id4 = p.addUserDebugText("TARGET", [center[0], center[1], center[2]+0.2], textColorRGB=[1,1,0], textSize=1.2, lifeTime=0, physicsClientId=cid)
        except TypeError:
            # Older pybullet may have different signature
            id4 = p.addUserDebugText("TARGET", [center[0], center[1], center[2]+0.2], [1,1,0], 1.2, physicsClientId=cid)

        self._target_marker_ids.extend([id1, id2, id3, id4])

    def _computeTerminated(self):
        """Computes the current done value."""
        return False

    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)
        
        # Truncate if drone goes out of bounds
        if (abs(state[0]) > 5 or abs(state[1]) > 5 or state[2] > 5 or state[2] < 0
                or abs(state[7]) > .4 or abs(state[8]) > .4):
            return True
        
        # Truncate if episode time exceeded
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        
        # Success condition: completed all target changes and stayed near final target
        if self.target_changes_completed >= self.num_target_changes:
            dist_to_final = np.linalg.norm(self.TARGET_POS - state[0:3])
            if dist_to_final < 0.2:
                return True
        
        return False
