"""
offset_exp.py: Training script for OffsetSphereAviary.
Starts from hover_into_sphere model and learns to hit an offset target.
"""

import cv2
import numpy as np
import torch as th
from datetime import datetime

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import pybullet as p

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Video
from stable_baselines3 import PPO

from OffsetSphereAviary import OffsetSphereAviary

# Camera settings - adjusted to see offset target
CAM_LOOKAT_POS = [0.15, 0.15, 0.5]  # Between drone start and target
CAM_POS = [1.2, 1.2, 1.5]

def get_frame(cid, target_pos, cam_pos):
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
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    
    _, _, rgb, depth, seg = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        physicsClientId=cid,
    )

    img = np.reshape(rgb, (h, w, 4))  # RGBA
    return img


class OffsetEvalCallback(BaseCallback):
    """Evaluates agent and tracks collision success rate."""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            self.evaluate_and_log()
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
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward

                # Capture frames for last episode
                if ep == self.n_eval_episodes - 1:
                    frame = get_frame(cid, CAM_LOOKAT_POS, CAM_POS)
                    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
                    ep_frames.append(frame)

            rewards.append(ep_reward)
            if terminated:  # Terminated means collision (success!)
                collisions += 1
            frames = ep_frames

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        collision_rate = collisions / self.n_eval_episodes

        if self.verbose > 0:
            print(f"Eval @ {self.num_timesteps}: reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                  f"collision_rate={collision_rate:.1%}")

        # Log to TensorBoard
        if self.logger:
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/collision_rate", collision_rate)
            self.logger.dump(self.num_timesteps)
            
            if len(frames) > 0:
                self.logger.record(
                    "trajectory/video",
                    Video(th.from_numpy(np.asarray([frames])), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
                print("logged video")

        return mean_reward, collision_rate


# Configuration
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])

# Target position: forward, right, and lower than [0,0,1]
TARGET_POS = [0.3, 0.3, 0.7]


def run(from_model="models/hover_into_sphere", total_timesteps=250000, n_envs=8):
    """
    Train on OffsetSphereAviary starting from hover_into_sphere model.
    """
    print("="*60)
    print("OFFSET SPHERE TRAINING")
    print(f"  Starting from: {from_model}")
    print(f"  Target position: {TARGET_POS}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Parallel envs: {n_envs}")
    print("="*60)

    # Create env factory
    def make_env(rank):
        def _init():
            env = OffsetSphereAviary(
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                target_pos=TARGET_POS,
            )
            return Monitor(env)
        return _init

    # Eval env
    eval_env = OffsetSphereAviary(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        target_pos=TARGET_POS,
    )
    
    print(f'[INFO] Action space: {eval_env.action_space}')
    print(f'[INFO] Observation space: {eval_env.observation_space}')
    print(f'[INFO] Target at: {TARGET_POS}')

    # Create parallel training environments
    sb3_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Load from hover_into_sphere model
    model = PPO.load(
        from_model,
        env=sb3_env,
        verbose=1,
        tensorboard_log="./ppo_offset_tensorboard/"
    )
    print(f"[INFO] Loaded base model from {from_model}")

    # Eval callback
    eval_callback = OffsetEvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=3,
        verbose=1
    )

    # Train!
    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_Offset"
    )

    # Save
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    model_path = f"models/offset_sphere_{timestamp}"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    sb3_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    # Start from hover_into_sphere, train to hit offset target
    run(from_model="models/hover_into_sphere", total_timesteps=150000, n_envs=5)
