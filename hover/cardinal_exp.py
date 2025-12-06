"""
Training script for CardinalHitAviary - randomized cardinal direction movement.
Trains until 5 million timesteps OR 8/8 perfect eval (2 per direction), whichever first.

NOTE: This environment has EXTENDED observations (18-dim instead of 12-dim) to include
target position and velocity. Cannot transfer from old models - must train from scratch!
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from CardinalHitAviary import CardinalHitAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

DEFAULT_ACT = ActionType('rpm')
INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])


class CardinalEvalCallback(BaseCallback):
    """
    Custom eval callback that tests 2 episodes per direction (8 total).
    Stops training early if 8/8 collisions achieved.
    """
    
    DIRECTIONS = ["right", "left", "forward", "back"]
    
    def __init__(self, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_collision_count = 0
        self.last_eval_timestep = 0
        self.perfect_achieved = False
        
    def _on_step(self):
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            
            print(f"\n{'='*60}")
            print(f"[EVAL @ {self.num_timesteps}] Testing 8 scenarios (2 per direction)")
            print(f"{'='*60}")
            
            total_collisions = 0
            total_reward = 0
            results_by_direction = {}
            
            # Test each direction twice
            for direction in self.DIRECTIONS:
                hits = 0
                dir_rewards = []
                
                for trial in range(2):
                    # Create eval env with fixed direction
                    eval_env = CardinalHitAviary(
                        act=DEFAULT_ACT,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        fixed_direction=direction,
                    )
                    
                    obs, _ = eval_env.reset()
                    done = False
                    ep_reward = 0
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        ep_reward += reward
                        done = terminated or truncated
                    
                    dir_rewards.append(ep_reward)
                    if info.get("collision_occurred", False):
                        hits += 1
                        total_collisions += 1
                    
                    eval_env.close()
                
                total_reward += sum(dir_rewards)
                results_by_direction[direction] = {
                    "hits": hits,
                    "avg_reward": np.mean(dir_rewards)
                }
                
                status = "✓✓" if hits == 2 else ("✓" if hits == 1 else "✗")
                print(f"  {direction.upper():>8}: {hits}/2 hits, avg reward: {np.mean(dir_rewards):.1f} [{status}]")
            
            mean_reward = total_reward / 8
            print(f"\n  TOTAL: {total_collisions}/8 collisions | Mean reward: {mean_reward:.1f}")
            
            if total_collisions > self.best_collision_count:
                self.best_collision_count = total_collisions
                self.model.save("models/cardinal_hit_best")
                print(f"  -> New best! ({total_collisions}/8) Saved to models/cardinal_hit_best")
            
            # Check for perfect score - early stopping!
            if total_collisions == 8:
                print(f"\n{'*'*60}")
                print(f"*** PERFECT 8/8! Training complete! ***")
                print(f"{'*'*60}\n")
                self.perfect_achieved = True
                self.model.save("models/cardinal_hit_perfect")
                return False  # Stop training!
            
            print(f"{'='*60}\n")
        
        return True


def train(total_timesteps=5_000_000, n_envs=5):
    """
    Train on CardinalHitAviary FROM SCRATCH.
    Extended observation space (18-dim) - cannot transfer from old models!
    Randomized cardinal directions - tests generalization!
    """
    print("="*60)
    print("CARDINAL HIT TRAINING - ALL DIRECTIONS!")
    print("  Training from SCRATCH (extended 18-dim observation space)")
    print(f"  Observation: drone state (12) + target pos (3) + target vel (3)")
    print(f"  Directions: right, left, forward, back (randomized)")
    print(f"  Starting positions: offset in direction of movement")
    print(f"  Speed: 6cm/s")
    print(f"  Timesteps: {total_timesteps} (or until 8/8 perfect)")
    print(f"  Parallel envs: {n_envs}")
    print("="*60)

    # Create env factory (random direction each episode)
    def make_env(rank):
        def _init():
            env = CardinalHitAviary(
                act=DEFAULT_ACT,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                # No fixed_direction = random each episode
            )
            return Monitor(env)
        return _init

    # Create parallel training environments
    sb3_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Create NEW model from scratch (can't transfer - different obs space!)
    model = PPO(
        "MlpPolicy",
        sb3_env,
        verbose=1,
        tensorboard_log="./ppo_cardinal_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
    )
    print(f"[INFO] Created NEW PPO model (18-dim observation space)")

    # Eval callback - tests all 4 directions
    eval_callback = CardinalEvalCallback(
        eval_freq=50000,  # Every 50k steps
        verbose=1
    )

    # Train!
    print(f"[INFO] Starting training for up to {total_timesteps} timesteps...")
    print(f"[INFO] Will stop early if 8/8 perfect eval achieved!")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_Cardinal"
    )

    # Save final
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    model_path = f"models/cardinal_hit_{timestamp}"
    model.save(model_path)
    
    if eval_callback.perfect_achieved:
        print(f"[INFO] Training stopped early - PERFECT 8/8!")
    else:
        print(f"[INFO] Training completed {total_timesteps} timesteps")
    
    print(f"[INFO] Final model saved to {model_path}")
    print(f"[INFO] Best model at models/cardinal_hit_best ({eval_callback.best_collision_count}/8)")

    sb3_env.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n_envs", type=int, default=5)
    args = parser.parse_args()
    
    train(total_timesteps=args.timesteps, n_envs=args.n_envs)
