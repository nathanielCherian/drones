import cv2
import time
import numpy as np
import torch as th
import pandas as pd
import argparse

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
from OffsetSphereAviary import OffsetSphereAviary
from ForceHitAviary import ForceHitAviary
from MovingHitAviary import MovingHitAviary
from CardinalHitAviary import CardinalHitAviary

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])

# Available environments
ENVS = {
    "tuned": TunedHoverAviary,
    "offset": OffsetSphereAviary,
    "force": ForceHitAviary,
    "moving": MovingHitAviary,
    "cardinal": CardinalHitAviary,
}

def run(env_name="tuned", model_path="models/hover_into_sphere"):

    # Select environment class
    if env_name not in ENVS:
        print(f"Unknown env '{env_name}'. Available: {list(ENVS.keys())}")
        return
    
    EnvClass = ENVS[env_name]
    print(f"[INFO] Using environment: {env_name} ({EnvClass.__name__})")
    print(f"[INFO] Loading model: {model_path}")

    env = EnvClass(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)


    model = PPO.load(model_path)

    obs, info = env.reset()
    done = False
    i = 0
    START = time.time()
    logs = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if truncated:
            print("TRUNCATED!")
        if terminated:
            force = info.get("collision_normal_force", 0)
            print(f"TERMINATED! (collision) - Normal force: {force:.4f} N")
        sync(i, START, env.CTRL_TIMESTEP)
        i += 1

        logs.append(list(obs[0][:3]) + [reward, terminated, truncated])

    df = pd.DataFrame(logs, columns=["x", "y", "z", "reward", "terminated", "truncated"])
    df.to_csv("log.csv", index=False)
    print(f"[INFO] Logged {len(logs)} steps to log.csv")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained model in an environment")
    parser.add_argument("--env", type=str, default="tuned", choices=["tuned", "offset", "force", "moving", "cardinal"],
                        help="Environment: 'tuned', 'offset', 'force', 'moving', or 'cardinal'")
    parser.add_argument("--model", type=str, default="models/hover_into_sphere",
                        help="Path to the model file (without .zip)")
    args = parser.parse_args()
    
    run(env_name=args.env, model_path=args.model)