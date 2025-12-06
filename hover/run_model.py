import cv2
import time
import numpy as np
import torch as th
import pandas as pd

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
import pybullet as p

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Video
from stable_baselines3 import PPO

from TunedHoverAviary import TunedHoverAviary

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

INIT_XYZS = np.array([[0, 0, 1]])
INIT_RPYS = np.array([[0, 0, 0]])

def run():

    env = TunedHoverAviary(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    # eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)


    model = PPO.load("models/ppo_hover_model_4d_3m_updated.zip")

    obs, info = env.reset()
    done = False
    i = 0
    START = time.time()
    
    # Limit log size to prevent unbounded memory growth
    MAX_LOG_STEPS = 50000
    logs = []
    
    while not done and len(logs) < MAX_LOG_STEPS:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if truncated:
            print("TRUNCATED!")
        sync(i, START, env.CTRL_TIMESTEP)
        i += 1

        # Get actual drone position and compute difference from target
        state = env._getDroneStateVector(0)
        # Get the true world position (unshifted). TunedHoverAviary overrides
        # `_getDroneStateVector` to return positions shifted relative to the target
        # for observations; call the parent class implementation to obtain the
        # absolute world-frame position.
        actual_pos = HoverAviary._getDroneStateVector(env, 0)[0:3]
        target_pos = env.TARGET_POS
        diff = actual_pos - target_pos
        
        # Print every 10 steps: observation (shifted), actual world position, and target
        observed_pos = obs[0][:3]  # observation vector is shifted so target appears at [0,0,1]
        if i % 10 == 0:
            print(
                f"Step {i} | ObsPos: ({observed_pos[0]:.2f}, {observed_pos[1]:.2f}, {observed_pos[2]:.2f}) | "
                f"WorldPos: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}) | "
                f"Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) | "
                f"Diff (dx, dy, dz): ({diff[0]:.2f}, {diff[1]:.2f}, {diff[2]:.2f})"
            )

        logs.append(list(obs[0][:3]) + [reward, terminated, truncated] + list(diff))

    df = pd.DataFrame(logs, columns=["x", "y", "z", "reward", "terminated", "truncated", "dx", "dy", "dz"])
    df.to_csv("log.csv", index=False)
    logs.clear()  # Free memory after saving


if __name__ == "__main__":
    run()