"""
col_exp_rpm.py: Training script for CollisionAviaryRPM.
PPO trains drone 0 with RAW RPM CONTROL to collide with drone 1 (PID-controlled).

This is the HARD version - the drone must learn to balance AND navigate.
"""

import cv2
import numpy as np
import torch as th
import os
import imageio

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import pybullet as p

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Video
from stable_baselines3 import PPO

from CollisionAviaryRPM import CollisionAviaryRPM

CAM_LOOKAT_POS = [0.5, 0.5, 0.5]  # Look between the two drones
CAM_POS = [2.5, 2.5, 2.0]         # Camera further back to see both drones

def get_frame(cid, target_pos, cam_pos):
    """Capture a frame from PyBullet for video logging."""
    view = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=[0, 0, 1],
        physicsClientId=cid,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=80.0,
        aspect=1.0,
        nearVal=0.05,
        farVal=10.0,
        physicsClientId=cid,
    )

    w, h = 720, 720
    _, _, rgb, depth, seg = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        physicsClientId=cid,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    img = np.reshape(rgb, (h, w, 4))  # RGBA
    return img


class CollisionEvalCallback(BaseCallback):
    """
    Evaluates the agent periodically and logs mean reward, collision stats, and videos.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, log_name="eval", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_name = log_name

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward, collision_rate = self.evaluate_and_log()
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                      f"collision_rate={collision_rate:.1%}")
        return True

    def evaluate_and_log(self):
        rewards = []
        collisions = 0
        frames = []
        cid = self.eval_env.getPyBulletClient()

        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_reward = 0
            ep_frames = []
            min_distance = float('inf')
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                
                # Track minimum distance achieved
                state0 = self.eval_env._getDroneStateVector(0)
                state1 = self.eval_env._getDroneStateVector(1)
                dist = np.linalg.norm(state1[0:3] - state0[0:3])
                min_distance = min(min_distance, dist)

                # Capture frame for video
                if self.logger:
                    frame = get_frame(cid, CAM_LOOKAT_POS, CAM_POS)
                    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
                    ep_frames.append(frame)

            # Print debug info
            print(f"  Ep {ep}: reward={ep_reward:.1f}, min_dist={min_distance:.3f}, "
                  f"drone0_final={state0[0:3]}, drone1_final={state1[0:3]}")

            rewards.append(ep_reward)
            if info.get("collision_occurred", False):
                collisions += 1
            
            # Keep frames from last episode for video
            frames = ep_frames

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        collision_rate = collisions / self.n_eval_episodes

        # Log scalar metrics
        if self.logger:
            self.logger.record(f"{self.log_name}/mean_reward", mean_reward)
            self.logger.record(f"{self.log_name}/std_reward", std_reward)
            self.logger.record(f"{self.log_name}/collision_rate", collision_rate)
            self.logger.dump(self.num_timesteps)
            
            # Log video to TensorBoard
            if len(frames) > 0:
                self.logger.record(
                    "trajectory/video",
                    Video(th.from_numpy(np.asarray([frames])), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                print("logged video to tensorboard")

        # Save video to disk as .mp4

        return mean_reward, std_reward, collision_rate


# Configuration - RAW RPM CONTROL
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')  # Raw motor control - harder to learn!


def run(from_model=None, total_timesteps=150000, n_envs=8):
    """
    Train PPO on CollisionAviaryRPM (raw motor control).
    
    Parameters
    ----------
    from_model : str, optional
        Path to a pre-trained model to continue training from.
    total_timesteps : int
        Total training timesteps.
    n_envs : int
        Number of parallel environments (default 8).
    """
    # Helper to create env instances
    def make_env(rank):
        def _init():
            env = CollisionAviaryRPM(
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                gui=False,
            )
            return Monitor(env)
        return _init

    # Create single env for eval and info display
    eval_env = CollisionAviaryRPM(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        gui=False,
    )
    
    print('[INFO] Action space:', eval_env.action_space)
    print('[INFO] Observation space:', eval_env.observation_space)
    print('[INFO] Using RAW RPM control - this is harder to learn!')
    print(f'[INFO] Using {n_envs} parallel environments!')

    # Create parallel training environments
    sb3_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Create or load model
    if from_model:
        model = PPO.load(from_model, env=sb3_env, verbose=1,
                         tensorboard_log="./ppo_collision_rpm_tensorboard/")
        print(f"[INFO] Loaded model from {from_model}")
    else:
        model = PPO("MlpPolicy", sb3_env, verbose=1,
                    tensorboard_log="./ppo_collision_rpm_tensorboard/")
        print("[INFO] Created new PPO model")

    # Create eval callback (uses separate eval_env, not the parallel ones)
    eval_callback = CollisionEvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=3,
        verbose=1
    )

    # Train!
    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name="PPO_Collision_RPM")

    # Save final model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_collision_rpm_{total_timesteps // 1000}k"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    sb3_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    # Train from scratch with raw RPM control and 8 parallel envs
    run(total_timesteps=150000, n_envs=8)
