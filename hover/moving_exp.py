"""
Training script for MovingHitAviary - moving target with mass for real force measurement.
Starts from collide_offset model (already knows how to hit a target).
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from MovingHitAviary import MovingHitAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])


class MovingEvalCallback(BaseCallback):
    """Custom eval callback that tracks collision force."""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_eval_timestep = 0
        
    def _on_step(self):
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            
            rewards = []
            forces = []
            collisions = 0
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                
                rewards.append(ep_reward)
                if info.get("collision_occurred", False):
                    collisions += 1
                    forces.append(info.get("max_force_this_episode", 0))
            
            mean_reward = np.mean(rewards)
            collision_rate = collisions / self.n_eval_episodes
            mean_force = np.mean(forces) if forces else 0
            
            print(f"\n[EVAL @ {self.num_timesteps}] Reward: {mean_reward:.1f} | "
                  f"Collisions: {collisions}/{self.n_eval_episodes} ({collision_rate*100:.0f}%) | "
                  f"Mean Force: {mean_force:.4f}N")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"models/moving_hit_best")
                print(f"  -> New best! Saved to models/moving_hit_best")
        
        return True


def train(from_model="models/collide_offset", total_timesteps=5_000_000, n_envs=5):
    """
    Train on MovingHitAviary starting from collide_offset model.
    The sphere has mass and moves slowly - learn to track and hit it hard!
    """
    print("="*60)
    print("MOVING HIT TRAINING - TRACK & HIT!")
    print(f"  Starting from: {from_model}")
    print(f"  Target: starts at [0,0,1], moves 2cm/s diagonally")
    print(f"  Sphere mass: 0.5kg (real force computation!)")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Parallel envs: {n_envs}")
    print("="*60)

    # Create env factory
    def make_env(rank):
        def _init():
            env = MovingHitAviary(
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
            )
            return Monitor(env)
        return _init

    # Eval env
    eval_env = MovingHitAviary(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
    )
    
    print(f'[INFO] Action space: {eval_env.action_space}')
    print(f'[INFO] Observation space: {eval_env.observation_space}')

    # Create parallel training environments
    sb3_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Load from collide_offset model (already knows how to hit targets!)
    model = PPO.load(
        from_model,
        env=sb3_env,
        verbose=1,
        tensorboard_log="./ppo_moving_tensorboard/"
    )
    print(f"[INFO] Loaded base model from {from_model}")

    # Eval callback
    eval_callback = MovingEvalCallback(
        eval_env,
        eval_freq=10000,
        n_eval_episodes=5,
        verbose=1
    )

    # Train!
    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_Moving"
    )

    # Save
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    model_path = f"models/moving_hit_{timestamp}"
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    sb3_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_model", type=str, default="models/collide_offset",
                        help="Base model to start from")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n_envs", type=int, default=5)
    args = parser.parse_args()
    
    train(from_model=args.from_model, total_timesteps=args.timesteps, n_envs=args.n_envs)
