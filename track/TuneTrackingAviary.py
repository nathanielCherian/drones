from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np

class TuneTrackingAviary(HoverAviary):
    """Tracking environment where target changes after reaching each waypoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 20  # Longer episode for multiple targets
        
        # Initialize with a fixed starting target (will change on reach)
        self.TARGET_POS = np.array([0, 0, 1])
        self.TARGET_RPY = np.array([0, 0, 0])
        
        # Store initial xyzs and rpys ranges for randomization on reset
        self._randomize_init_pos = True
        self._randomize_target_pos = True
        
        # Generate a sequence of waypoints for tracking
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        self.TARGET_POS = self.waypoints[self.current_waypoint_idx]
        
        # Distance threshold to consider target "reached"
        self.target_reached_threshold = 0.2

    def _generate_waypoints(self, num_waypoints=5):
        """Generate a sequence of random waypoints for the drone to visit."""
        waypoints = []
        for _ in range(num_waypoints):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0.3, 1.5)  # Keep z between 0.3 and 1.5
            waypoints.append(np.array([x, y, z]))
        return waypoints

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
        else:
            self.INIT_RPYS = np.array([[0, 0, 0]])
            self.INIT_XYZS = np.array([[0, 0, 1]])
        
        # Reset waypoint tracking
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        self.TARGET_POS = self.waypoints[self.current_waypoint_idx]
        
        print(f"Reset: Initial target waypoint {self.current_waypoint_idx}: {self.TARGET_POS}")
        # Call parent reset with any kwargs (e.g., seed)
        return super().reset(**kwargs)

    def _computeReward(self):
        """Computes the current reward value.

        Combines penalties for distance and angle difference with bonuses for
        reaching waypoints and moving through the sequence.

        Returns
        -------
        float
            The reward.

        """
        # Acquire state and velocity
        state = self._getDroneStateVector(0)
        # Note: in this environment velocity components are at indices 10:13
        vel = state[10:13]
        pos = state[0:3]

        # Distance to current target
        norm = np.linalg.norm(self.TARGET_POS - pos)

        # Base distance reward (linear with distance)
        distance_reward = max(0.0, 4.0 - norm)

        # Vertical offset relative to target (positive means above target)
        dz = pos[2] - self.TARGET_POS[2]

        # Reward motion toward the target altitude (non-quadratic):
        # term = -k * dz * vz  => positive when velocity is directed toward target
        vz = float(vel[2])
        vel_towards_reward = -2.0 * dz * vz
        vel_towards_reward = float(np.clip(vel_towards_reward, -3.0, 3.0))

        # Penalize overall speed to encourage smooth hovering
        speed = float(np.linalg.norm(vel))
        speed_penalty = 0.2 * speed

        # Bonus for being very close and nearly stationary
        close_bonus = 0.0
        if norm < 0.2 and speed < 0.2:
            close_bonus = 1.5

        # Small stabilizing bonus when near target (encourages low velocity)
        near_vel_bonus = 0.0
        if norm < 0.2:
            near_vel_bonus = max(0.0, 1.0 - speed) * 2.0
        #print(norm)
        # Bonus for completing waypoints (progressing through sequence)
        waypoint_bonus = 0.0
        if norm < self.target_reached_threshold:
            # Waypoint reached! Change to next one.
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
                self.TARGET_POS = self.waypoints[self.current_waypoint_idx]
                waypoint_bonus = 5.0  # Large bonus for reaching waypoint
                print(f"Waypoint {self.current_waypoint_idx - 1} reached! Moving to waypoint {self.current_waypoint_idx}: {self.TARGET_POS}")
            else:
                # All waypoints completed
                waypoint_bonus = 10.0
                print(f"All waypoints completed!")

        # Combine terms
        ret = distance_reward + vel_towards_reward + waypoint_bonus

        return float(ret)

    def _getDroneStateVector(self, drone_id: int):
        """Return drone state vector with positions relative to target.
        
        Shifts position so the target appears at [0, 0, 1].
        This makes the observation invariant to absolute position, helping
        generalization to new target locations.
        """
        state = super()._getDroneStateVector(drone_id).copy()
        
        # Position is at indices 0:3
        # Shift position so target is at [0, 0, 1] in the drone's frame
        target_offset = self.TARGET_POS - np.array([0, 0, 1])
        state[0:3] = state[0:3] - target_offset
        
        return state

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        return False

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 4 or abs(state[1]) > 4 or state[2] > 10 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
