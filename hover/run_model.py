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
from HoverV2Aviary import HoverV2Aviary
from OffcenterV2Aviary import OffcenterV2Aviary
from MovingV2Aviary import MovingV2Aviary
from CardinalV2Aviary import CardinalV2Aviary

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
    # V2 environments (18-dim obs with target pos/vel)
    "hover_v2": HoverV2Aviary,
    "offcenter_v2": OffcenterV2Aviary,
    "moving_v2": MovingV2Aviary,
    "cardinal_v2": CardinalV2Aviary,
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


def run_cardinal_demo(model_path, env_type="cardinal_v2"):
    """Run all 4 cardinal directions in sequence."""
    DIRECTIONS = ["right", "left", "forward", "back"]
    
    # Select the right cardinal env class
    if env_type == "cardinal_v2":
        EnvClass = CardinalV2Aviary
    else:
        EnvClass = CardinalHitAviary
    
    print(f"\n{'='*60}")
    print(f"CARDINAL DEMO - All 4 directions")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")
    
    model = PPO.load(model_path)
    results = []
    
    for direction in DIRECTIONS:
        print(f"\n>>> Direction: {direction.upper()}")
        print("-" * 40)
        
        env = EnvClass(
            gui=True, 
            act=DEFAULT_ACT, 
            initial_xyzs=INIT_XYZS, 
            initial_rpys=INIT_RPYS,
            fixed_direction=direction,
        )
        
        obs, info = env.reset()
        done = False
        i = 0
        START = time.time()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            sync(i, START, env.CTRL_TIMESTEP)
            i += 1
        
        hit = info.get("collision_occurred", False)
        results.append({"direction": direction, "hit": hit, "steps": i})
        
        if hit:
            print(f"  ✓ HIT! ({i} steps)")
        else:
            print(f"  ✗ MISS (timeout after {i} steps)")
        
        env.close()
        time.sleep(0.5)  # Brief pause between runs
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    hits = sum(1 for r in results if r["hit"])
    for r in results:
        status = "✓" if r["hit"] else "✗"
        print(f"  {r['direction'].upper():>8}: {status}")
    print(f"\nTotal: {hits}/4 directions hit")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained model in an environment")
    parser.add_argument("--env", type=str, default="tuned", 
                        choices=["tuned", "offset", "force", "moving", "cardinal",
                                 "hover_v2", "offcenter_v2", "moving_v2", "cardinal_v2"],
                        help="Environment name (v2 versions have extended observations)")
    parser.add_argument("--model", type=str, default="models/hover_into_sphere",
                        help="Path to the model file (without .zip)")
    parser.add_argument("--demo", action="store_true",
                        help="For cardinal envs: run all 4 directions automatically")
    args = parser.parse_args()
    
    if args.demo and args.env in ["cardinal", "cardinal_v2"]:
        run_cardinal_demo(args.model, env_type=args.env)
    else:
        run(env_name=args.env, model_path=args.model)