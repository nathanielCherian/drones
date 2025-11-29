import time
import numpy as np
import pandas as pd

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3 import PPO

from TrackingAviary import TrackingAviary

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

INIT_XYZS = np.array([[0, 0, 1]])
INIT_RPYS = np.array([[0, 0, 0]])


def run(model_path="models/ppo_tracking_model.zip", gui=True):
    """
    Run a trained tracking model in the environment.
    
    Args:
        model_path: Path to the trained PPO model
        gui: Whether to display PyBullet GUI
    """
    # Create environment with tracking parameters
    env = TrackingAviary(
        num_target_changes=7,
        target_change_distance=1.0,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        gui=gui
    )
    
    print(f'[INFO] Action space: {env.action_space}')
    print(f'[INFO] Observation space: {env.observation_space}')

    # Load trained model
    model = PPO.load(model_path, device="cpu")
    print(f'[INFO] Loaded model from {model_path}')

    # Run episode
    obs, info = env.reset()
    done = False
    i = 0
    START = time.time()
    logs = []
    
    print(f'[INFO] Starting episode with target at {env.TARGET_POS}')

    while not done:
        # Predict action with deterministic policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Sync with real-time (optional)
        # from gym_pybullet_drones.utils.utils import sync
        # sync(i, START, env.CTRL_TIMESTEP)
        
        i += 1

        # Log state, reward, target info
        logs.append({
            'timestep': i,
            'x': obs[0, 0],
            'y': obs[0, 1],
            'z': obs[0, 2],
            'target_x': env.TARGET_POS[0],
            'target_y': env.TARGET_POS[1],
            'target_z': env.TARGET_POS[2],
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'targets_completed': env.target_changes_completed,
        })

        # Print target change info
        if i > 1 and env.target_changes_completed > logs[-2]['targets_completed']:
            print(f'[INFO] Target {env.target_changes_completed} reached! New target: {env.TARGET_POS}')

    # Save logs to CSV
    df = pd.DataFrame(logs)
    df.to_csv("tracking_log.csv", index=False)
    print(f'\n[INFO] Episode finished after {i} timesteps')
    print(f'[INFO] Targets completed: {env.target_changes_completed} / {env.num_target_changes}')
    print(f'[INFO] Logs saved to tracking_log.csv')

    env.close()


if __name__ == "__main__":
    # Run with GUI (can be slow for real-time visualization)
    run(model_path="models/ppo_tracking_model.zip", gui=True)
    
    # Or run without GUI for faster execution:
    # run(model_path="models/ppo_tracking_model.zip", gui=False)
