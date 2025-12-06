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

from TuneTrackingAviary import TuneTrackingAviary
from TrackingAviary import TrackingAviary

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

INIT_XYZS = np.array([[0, 0, 1]])
INIT_RPYS = np.array([[0, 0, 0]])

def run():

    env = TrackingAviary(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
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

        # Get absolute world position from parent class
        world_state = HoverAviary._getDroneStateVector(env, 0)
        world_pos = world_state[0:3]
        obs_pos = obs[0][:3]  # Shifted observation
        target_pos = env.TARGET_POS
        dist_to_target = np.linalg.norm(target_pos - world_pos)
        
        # Print every 10 steps
        if i % 10 == 0:
            print(f"Step {i:5d} | ObsPos: [{obs_pos[0]:6.3f}, {obs_pos[1]:6.3f}, {obs_pos[2]:6.3f}] | "
                  f"WorldPos: [{world_pos[0]:6.3f}, {world_pos[1]:6.3f}, {world_pos[2]:6.3f}] | "
                  f"Target: [{target_pos[0]:6.3f}, {target_pos[1]:6.3f}, {target_pos[2]:6.3f}] | "
                  f"Distance: {dist_to_target:6.3f} | Reward: {reward:7.3f}")

        logs.append(list(obs_pos) + list(world_pos) + list(target_pos) + [dist_to_target, reward, terminated, truncated, env.target_changes_completed])

    df = pd.DataFrame(logs, columns=["obs_x", "obs_y", "obs_z", "world_x", "world_y", "world_z", "target_x", "target_y", "target_z", "distance", "reward", "terminated", "truncated", "target_changes"])
    #df.to_csv("tracking_log.csv", index=False)
    logs.clear()  # Free memory after saving


if __name__ == "__main__":
    run()
