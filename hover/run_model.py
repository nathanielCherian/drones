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
from TrueRandomAviary import TrueRandomAviary

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
    "true_random": TrueRandomAviary,
}

def run(env_name="tuned", model_path="models/hover_into_sphere", fixed_speed=None):

    # Select environment class
    if env_name not in ENVS:
        print(f"Unknown env '{env_name}'. Available: {list(ENVS.keys())}")
        return
    
    EnvClass = ENVS[env_name]
    print(f"[INFO] Using environment: {env_name} ({EnvClass.__name__})")
    print(f"[INFO] Loading model: {model_path}")

    # Build env kwargs
    env_kwargs = dict(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    
    # Pass fixed_speed for true_random env and calculate corresponding bounds
    if env_name == "true_random" and fixed_speed is not None:
        env_kwargs["fixed_speed"] = fixed_speed
        
        # Calculate bounds based on training curriculum
        # Training: starts at min_speed=0.02, increments by 0.002 each stage
        # Bounds: starts at max_xy=0.5, max_z=0.8, increments by 0.1 each stage
        INITIAL_MIN_SPEED = 0.02
        SPEED_INCREMENT = 0.002
        INITIAL_MAX_XY = 0.5
        INITIAL_MAX_Z = 0.8
        BOUNDS_INCREMENT = 0.1
        MAX_Z_CAP = 3.0
        
        # How many stages to reach this speed as min_speed?
        stages = max(0, int((fixed_speed - INITIAL_MIN_SPEED) / SPEED_INCREMENT))
        max_xy = INITIAL_MAX_XY + stages * BOUNDS_INCREMENT
        max_z = min(INITIAL_MAX_Z + stages * BOUNDS_INCREMENT, MAX_Z_CAP)
        
        env_kwargs["max_xy"] = max_xy
        env_kwargs["max_z"] = max_z
        
        print(f"[INFO] Fixed speed: {fixed_speed*100:.1f} cm/s")
        print(f"[INFO] Calculated bounds (stage {stages}): XY=±{max_xy:.2f}m, Z=[0.4, {max_z:.2f}]m")
    
    env = EnvClass(**env_kwargs)
    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)

    model = PPO.load(model_path)

    # For true_random: loop forever until Ctrl+C
    if env_name == "true_random":
        print("\n[INFO] Running in loop mode. Press Ctrl+C to exit.\n")
        print("[INFO] Camera tracks drone. Use mouse to rotate view.\n")
        episode = 0
        try:
            while True:
                episode += 1
                obs, info = env.reset()
                done = False
                i = 0
                START = time.time()
                
                target_pos = info.get("target_pos", env.TARGET_POS)
                print(f"--- Episode {episode} | Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] ---")
                
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Camera tracking: follow drone position
                    drone_pos = env._getDroneStateVector(0)[0:3]
                    target_pos = env.TARGET_POS
                    # Look at midpoint between drone and target
                    cam_target = (drone_pos + target_pos) / 2
                    # Distance based on separation
                    dist = np.linalg.norm(drone_pos - target_pos)
                    cam_dist = max(3.0, dist * 1.5)  # At least 3m, or 1.5x the separation
                    p.resetDebugVisualizerCamera(
                        cameraDistance=cam_dist,
                        cameraYaw=45,
                        cameraPitch=-30,
                        cameraTargetPosition=cam_target.tolist(),
                        physicsClientId=env.CLIENT
                    )
                    
                    sync(i, START, env.CTRL_TIMESTEP)
                    i += 1
                
                top_speed = info.get("top_drone_speed", 0)
                if terminated:
                    force = info.get("collision_normal_force", 0)
                    print(f"  -> HIT! (step {i}) - Normal force: {force:.4f} N | Top drone speed: {top_speed:.2f} m/s")
                else:
                    print(f"  -> MISS (truncated at step {i}) | Top drone speed: {top_speed:.2f} m/s")
                    
        except KeyboardInterrupt:
            print(f"\n[INFO] Stopped after {episode} episodes.")
        
        env.close()
        return

    # Original single-run behavior for other envs
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

        # Handle both 1D (18,) and 2D (1, 18) observation shapes
        obs_flat = obs.flatten() if hasattr(obs, 'flatten') else obs
        logs.append(list(obs_flat[:3]) + [reward, terminated, truncated])

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
                                 "hover_v2", "offcenter_v2", "moving_v2", "cardinal_v2",
                                 "true_random"],
                        help="Environment name (v2 versions have extended observations)")
    parser.add_argument("--model", type=str, default="models/hover_into_sphere",
                        help="Path to the model file (without .zip)")
    parser.add_argument("--demo", action="store_true",
                        help="For cardinal envs: run all 4 directions automatically")
    parser.add_argument("--speed", type=float, default=None,
                        help="Fixed speed in cm/s for true_random env (e.g. --speed 10)")
    args = parser.parse_args()
    
    # Convert cm/s to m/s
    fixed_speed = args.speed / 100.0 if args.speed is not None else None
    
    if args.demo and args.env in ["cardinal", "cardinal_v2"]:
        run_cardinal_demo(args.model, env_type=args.env)
    else:
        run(env_name=args.env, model_path=args.model, fixed_speed=fixed_speed)