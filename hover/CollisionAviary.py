"""
CollisionAviary: Two-drone environment where PPO controls one drone
and the other hovers at a fixed point using a PID controller.
Reward is based on distance to the stationary drone + bonus for collision.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CollisionAviary(BaseRLAviary):
    """Two-drone env: PPO controls drone 0, PID holds drone 1 stationary."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,  # VEL is much easier to learn
                 ):
        """
        Parameters
        ----------
        See HoverAviary for details. Key difference: 2 drones.
        """
        # Target position for the stationary drone (drone 1)
        self.TARGET_DRONE_POS = np.array([1.0, 1.0, 1.0])
        
        # Initial positions: drone 0 on ground, drone 1 at target
        initial_xyzs = np.array([
            [0.0, 0.0, 0.05],       # PPO-controlled drone starts near ground
            self.TARGET_DRONE_POS   # PID-controlled drone starts at hover target
        ])
        initial_rpys = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        self.EPISODE_LEN_SEC = 10

        # Initialize base with 2 drones
        super().__init__(
            drone_model=drone_model,
            num_drones=2,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )

        # Create PID controller for drone 1 (the stationary one)
        self.pid_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        # Track if collision happened this episode (for logging)
        self.collision_occurred = False
        self.total_collision_force = 0.0

    ################################################################################

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.collision_occurred = False
        self.total_collision_force = 0.0
        self.pid_ctrl.reset()
        return super().reset(seed=seed, options=options)

    ################################################################################

    def step(self, action):
        """
        Step the environment.
        
        action: The PPO action for drone 0 only (shape depends on ActionType).
        We compute PID action for drone 1 internally to keep it stationary.
        """
        # Get drone 1's current state for PID control
        state1 = self._getDroneStateVector(1)
        pos1 = state1[0:3]
        quat1 = state1[3:7]
        vel1 = state1[10:13]
        ang_vel1 = state1[13:16]

        # Compute PID RPMs to hold drone 1 at TARGET_DRONE_POS
        rpm1, _, _ = self.pid_ctrl.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=pos1,
            cur_quat=quat1,
            cur_vel=vel1,
            cur_ang_vel=ang_vel1,
            target_pos=self.TARGET_DRONE_POS,
            target_rpy=np.zeros(3),
            target_vel=np.zeros(3),
            target_rpy_rates=np.zeros(3)
        )

        # Normalize PID rpm to [-1, 1] for drone 1
        rpm1_clipped = np.clip(rpm1, 0, self.MAX_RPM)
        normalized_rpm1 = (rpm1_clipped / self.MAX_RPM) * 2 - 1

        # First, preprocess the PPO action for drone 0
        ppo_action = np.array(action).reshape(1, -1)

        # Build combined action array based on action type
        if self.ACT_TYPE == ActionType.VEL:
            # For VEL: drone 0 gets PPO velocity command (vx, vy, vz, yaw_rate)
            # drone 1: we want zero velocity (hover in place)
            # VEL action is (NUM_DRONES, 4) where each row is [vx, vy, vz, yaw_rate] normalized
            drone1_vel_action = np.array([[0.0, 0.0, 0.0, 0.0]])  # Zero velocity = hover
            full_action = np.vstack([ppo_action, drone1_vel_action])
        elif self.ACT_TYPE == ActionType.RPM:
            full_action = np.vstack([ppo_action, normalized_rpm1.reshape(1, 4)])
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            ppo_4motors = np.tile(ppo_action, (1, 4))
            full_action = np.vstack([ppo_4motors, normalized_rpm1.reshape(1, 4)])
        else:
            full_action = np.vstack([ppo_action, normalized_rpm1.reshape(1, -1)])

        # Call parent step with combined action
        obs, reward, terminated, truncated, info = super().step(full_action)

        return obs, reward, terminated, truncated, info

    ################################################################################

    def _actionSpace(self):
        """Returns the action space for drone 0 only (PPO-controlled)."""
        # We only expose action space for drone 0
        if self.ACT_TYPE == ActionType.RPM:
            # 4 motors, normalized [-1, 1]
            return spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return spaces.Box(low=-1.0, high=1.0, shape=(1, 1), dtype=np.float32)
        elif self.ACT_TYPE == ActionType.VEL:
            # Velocity control: vx, vy, vz, yaw_rate (normalized)
            return spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)
        else:
            # Default to 4D
            return spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)

    ################################################################################

    def _observationSpace(self):
        """Returns observation space for drone 0 only."""
        # Use parent's observation space but only for 1 drone
        # We'll return drone 0's obs in _computeObs
        if self.OBS_TYPE == ObservationType.KIN:
            # Kinematic obs: pos(3) + quat(4) + rpy(3) + vel(3) + ang_vel(3) = 16
            # Plus: last_action(4*ACTION_BUFFER_SIZE)
            # Plus: relative position to target drone (3) + relative velocity (3) = 6
            # Total = 16 + 4*ACTION_BUFFER_SIZE + 6
            lo = -np.inf
            hi = np.inf
            base_low = [lo, lo, 0] + [lo]*4 + [lo]*3 + [lo]*3 + [lo]*3  # pos, quat, rpy, vel, ang_vel = 16
            base_high = [hi]*16
            action_low = [-1.0] * (4 * self.ACTION_BUFFER_SIZE)
            action_high = [1.0] * (4 * self.ACTION_BUFFER_SIZE)
            rel_low = [lo]*6  # rel_pos(3) + rel_vel(3)
            rel_high = [hi]*6
            
            obs_lower = np.array([base_low + action_low + rel_low], dtype=np.float32)
            obs_upper = np.array([base_high + action_high + rel_high], dtype=np.float32)
            return spaces.Box(low=obs_lower, high=obs_upper, dtype=np.float32)
        else:
            # For RGB or other, use parent
            return super()._observationSpace()

    ################################################################################

    def _computeObs(self):
        """Compute observation for drone 0, including info about drone 1."""
        if self.OBS_TYPE == ObservationType.KIN:
            # Get base kinematic obs for drone 0
            state0 = self._getDroneStateVector(0)
            state1 = self._getDroneStateVector(1)
            
            # Drone 0's state: pos, quat, rpy, vel, ang_vel
            obs_kin = np.hstack([
                state0[0:3],    # pos
                state0[3:7],    # quat  
                state0[7:10],   # rpy
                state0[10:13],  # vel
                state0[13:16]   # ang_vel
            ]).reshape(1, -1)

            # Add action buffer
            act_buffer = np.zeros((1, 4 * self.ACTION_BUFFER_SIZE))
            for i, act in enumerate(self.action_buffer):
                act_buffer[0, i*4:(i+1)*4] = act[0, :4] if act.shape[1] >= 4 else np.zeros(4)

            # Relative info about target drone
            rel_pos = state1[0:3] - state0[0:3]  # Position of drone 1 relative to drone 0
            rel_vel = state1[10:13] - state0[10:13]  # Relative velocity

            obs = np.hstack([obs_kin, act_buffer, rel_pos.reshape(1, 3), rel_vel.reshape(1, 3)])
            return obs.astype(np.float32)
        else:
            return super()._computeObs()

    ################################################################################

    def _computeReward(self):
        """
        Reward based on:
        1. Distance from drone 0 to drone 1 (closer = better)
        2. Large bonus for collision contact
        3. Stability (not too tilted)
        """
        state0 = self._getDroneStateVector(0)
        state1 = self._getDroneStateVector(1)

        pos0 = state0[0:3]
        pos1 = state1[0:3]

        # Distance to target drone
        distance = np.linalg.norm(pos1 - pos0)

        # Reward: higher constant, squared distance (not ^4)
        # At 1.7m: 4 - 2.89 = 1.11, At 1m: 3, At 0m: 4
        # Still steeper gradient as it gets closer
        reward = max(0, 4 - distance**2)

        # Check for collision using PyBullet contact points
        contacts = p.getContactPoints(
            bodyA=self.DRONE_IDS[0],
            bodyB=self.DRONE_IDS[1],
            physicsClientId=self.CLIENT
        )

        collision_force = 0.0
        if len(contacts) > 0:
            for c in contacts:
                collision_force += abs(c[9])  # Normal force magnitude
            
            # Large reward for making contact!
            reward += 50.0  # Big bonus for collision
            
            self.collision_occurred = True
            self.total_collision_force += collision_force

        return float(reward)

    ################################################################################

    def _computeTerminated(self):
        """Episode terminates on collision (success!)."""
        # Check for collision
        contacts = p.getContactPoints(
            bodyA=self.DRONE_IDS[0],
            bodyB=self.DRONE_IDS[1],
            physicsClientId=self.CLIENT
        )
        if len(contacts) > 0:
            return True
        return False

    ################################################################################

    def _computeTruncated(self):
        """Truncate if drone 0 goes out of bounds or times out."""
        state0 = self._getDroneStateVector(0)

        # Out of bounds check
        if (abs(state0[0]) > 3.0 or abs(state0[1]) > 3.0 or state0[2] > 4.0 or state0[2] < 0.0):
            return True

        # Too tilted
        if abs(state0[7]) > 0.6 or abs(state0[8]) > 0.6:
            return True

        # Time limit
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ################################################################################

    def _computeInfo(self):
        """Return info dict with collision data."""
        return {
            "collision_occurred": self.collision_occurred,
            "total_collision_force": self.total_collision_force,
        }
