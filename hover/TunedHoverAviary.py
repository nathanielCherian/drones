from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import numpy as np

class TunedHoverAviary(HoverAviary):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 10
        #self.TARGET_POS = np.array([0, 0, 1])
        
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)
        # Keep TARGET_POS as a 1D array [x, y, z] to match usage elsewhere
        self.TARGET_POS = np.array([0, 0, 1])
        self.TARGET_RPY = np.array([0, 0, 0])
        # Store initial xyzs and rpys ranges for randomization on reset
        self._randomize_init_pos = True
        self._randomize_target_pos = True
        # Curriculum parameters: gradually increase randomization range over time
        self._global_step_count = 0
        # Number of environment steps to reach full difficulty
        self.curriculum_timesteps = 200_000
        # start and end radii for target randomization (meters)
        self.curriculum_start_radius = 0.15
        self.curriculum_end_radius = 1.2
        # start and end radii for initial position randomization
        self.init_start_radius = 0.1
        self.init_end_radius = 1.0

    def reset(self, **kwargs):
        """Reset the environment and randomize initial position if enabled."""
        if self._randomize_init_pos:
            # Interpolate initial position radius by curriculum progress
            prog = min(self._global_step_count / max(1, self.curriculum_timesteps), 1.0)
            init_radius = self.init_start_radius + prog * (self.init_end_radius - self.init_start_radius)
            # Randomize x, y in [-init_radius, init_radius] and z in [0.5, 2]
            x = np.random.uniform(-init_radius, init_radius)
            y = np.random.uniform(-init_radius, init_radius)
            z = np.random.uniform(0.5, 2.0)
            self.INIT_XYZS = np.array([[x, y, z]])
            # Keep rotations at zero
            self.INIT_RPYS = np.array([[0, 0, 0]])
        else:
            self.INIT_RPYS = np.array([[0, 0, 0]])
            self.INIT_XYZS = np.array([[0, 0, 1]])
        if self._randomize_target_pos:
            # Curriculum progress (0 -> start, 1 -> full difficulty)
            prog = min(self._global_step_count / max(1, self.curriculum_timesteps), 1.0)
            radius = self.curriculum_start_radius + prog * (self.curriculum_end_radius - self.curriculum_start_radius)
            # Sample a random offset within a sphere of 'radius' around nominal target z=1
            # Sample direction uniformly on sphere and radius uniform in [0, radius]
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            r = np.random.uniform(0, radius)
            dx = r * np.cos(phi) * np.cos(theta)
            dy = r * np.cos(phi) * np.sin(theta)
            dz = r * np.sin(phi)
            new_x = dx
            new_y = dy
            new_z = np.clip(1.0 + dz, 0.5, 2.5)
            self.TARGET_POS = np.array([new_x, new_y, new_z])
        #print("target pos", self.TARGET_POS)
        # Call parent reset with any kwargs (e.g., seed)
        return super().reset(**kwargs)

    def _computeReward(self):
        """Computes the current reward value.

        Combines penalties for distance and angle difference with bonuses for
        reaching the target position.

        Returns
        -------
        float
            The reward.

        """
        # Acquire RAW (absolute) state and velocity from parent so reward
        # calculations are done in the world frame relative to TARGET_POS.
        # Note: _getDroneStateVector() below shifts positions to make the
        # target appear at [0,0,1] (see override). To compute correct
        # distances to an arbitrary TARGET_POS we must use the parent's
        # unshifted state vector here.
        raw_state = super()._getDroneStateVector(0).copy()
        # Note: velocity components are at indices 10:13 in the raw state
        vel = raw_state[10:13]
        pos = raw_state[0:3]

        # Distance to target (use absolute positions)
        norm = np.linalg.norm(self.TARGET_POS - pos)

        # Base distance reward (linear with distance)
        distance_reward = max(0.0, 4.0 - norm)

        # Vertical offset relative to target (positive means above target)
        # Offsets relative to target (positive means drone is greater than target coordinate)
        dx = pos[0] - self.TARGET_POS[0]
        dy = pos[1] - self.TARGET_POS[1]
        dz = pos[2] - self.TARGET_POS[2]

        # Reward motion toward the target along each axis (non-quadratic):
        # term_axis = -k_axis * d_axis * v_axis => positive when velocity reduces the error
        vx = float(vel[0])
        vy = float(vel[1])
        vz = float(vel[2])

        kx, ky, kz = 1.5, 1.5, 2.0
        vx_term = -kx * dx * vx
        vy_term = -ky * dy * vy
        vz_term = -kz * dz * vz

        # Combine and clip to avoid large spikes
        vel_towards_reward = float(np.clip(vx_term + vy_term + vz_term, -7.0, 7.0))

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

        # Combine terms
        ret = distance_reward + vel_towards_reward
        #ret = distance_reward + vel_towards_reward + close_bonus + near_vel_bonus - speed_penalty

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
        # state = self._getDroneStateVector(0)
        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
        #     return True
        # else:
        #     return False


    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 2 or abs(state[1]) > 2 or state[2] > 5 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def step(self, action):
        """Override step to increment global step counter for curriculum progression."""
        # Increment per-environment global step counter
        try:
            self._global_step_count += 1
        except Exception:
            self._global_step_count = 1
        return super().step(action)