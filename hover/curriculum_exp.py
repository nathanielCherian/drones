"""
Curriculum Learning Training Script
====================================
Trains through 4 progressive stages, each building on the previous model.

Stage 1: HoverV2 - Hover up to stationary sphere at [0,0,1]
Stage 2: OffcenterV2 - Stationary sphere offset at [0.3, 0.3, 0.7]
Stage 3: MovingV2 - Moving sphere (fixed direction)
Stage 4: CardinalV2 - Randomized cardinal directions

Each stage:
- Trains up to 5M timesteps
- Evaluates 10 times (except Stage 4 which has 8 tests: 2 per direction)
- Moves on when 10/10 (or 8/8) perfect
- Saves model as curriculum_part_<stage_name>

All environments have extended 18-dim observations:
[drone_pos(3), drone_rpy(3), drone_vel(3), drone_ang_vel(3), target_pos(3), target_vel(3)]
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

from HoverV2Aviary import HoverV2Aviary
from OffcenterV2Aviary import OffcenterV2Aviary
from MovingV2Aviary import MovingV2Aviary
from CardinalV2Aviary import CardinalV2Aviary
from gym_pybullet_drones.utils.enums import ActionType

DEFAULT_ACT = ActionType('rpm')
INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])

# Curriculum stages configuration
STAGES = [
    {
        "name": "hover",
        "env_class": HoverV2Aviary,
        "n_eval": 10,
        "description": "Stationary sphere at [0, 0, 1]",
        "color": "Green",
    },
    {
        "name": "offcenter",
        "env_class": OffcenterV2Aviary,
        "n_eval": 10,
        "description": "Stationary sphere at [0.3, 0.3, 0.7]",
        "color": "Yellow",
    },
    {
        "name": "moving",
        "env_class": MovingV2Aviary,
        "n_eval": 10,
        "description": "Moving sphere (fixed direction, 6cm/s)",
        "color": "Blue",
    },
    {
        "name": "cardinal",
        "env_class": CardinalV2Aviary,
        "n_eval": 8,  # 2 per direction x 4 directions
        "description": "Randomized cardinal directions",
        "color": "Orange",
        "is_cardinal": True,
    },
]


class CurriculumEvalCallback(BaseCallback):
    """
    Eval callback for curriculum learning.
    Stops training when perfect score achieved.
    """
    
    def __init__(self, env_class, n_eval, eval_freq=50000, is_cardinal=False, verbose=1):
        super().__init__(verbose)
        self.env_class = env_class
        self.n_eval = n_eval
        self.eval_freq = eval_freq
        self.is_cardinal = is_cardinal
        self.best_collision_count = 0
        self.last_eval_timestep = 0
        self.perfect_achieved = False
        
    def _on_step(self):
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            
            print(f"\n{'='*60}")
            print(f"[EVAL @ {self.num_timesteps}] Testing {self.n_eval} scenarios")
            print(f"{'='*60}")
            
            if self.is_cardinal:
                collisions = self._eval_cardinal()
            else:
                collisions = self._eval_standard()
            
            if collisions > self.best_collision_count:
                self.best_collision_count = collisions
                print(f"  -> New best! ({collisions}/{self.n_eval})")
            
            if collisions == self.n_eval:
                print(f"\n{'*'*60}")
                print(f"*** PERFECT {self.n_eval}/{self.n_eval}! Stage complete! ***")
                print(f"{'*'*60}\n")
                self.perfect_achieved = True
                return False  # Stop training
            
            print(f"{'='*60}\n")
        
        return True
    
    def _eval_standard(self):
        """Standard evaluation: n_eval episodes with random seed."""
        collisions = 0
        rewards = []
        
        for i in range(self.n_eval):
            eval_env = self.env_class(
                act=DEFAULT_ACT,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
            )
            
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            rewards.append(ep_reward)
            if info.get("collision_occurred", False):
                collisions += 1
            
            eval_env.close()
        
        print(f"  Collisions: {collisions}/{self.n_eval} | Mean reward: {np.mean(rewards):.1f}")
        return collisions
    
    def _eval_cardinal(self):
        """Cardinal evaluation: 2 trials per direction."""
        DIRECTIONS = ["right", "left", "forward", "back"]
        total_collisions = 0
        
        for direction in DIRECTIONS:
            hits = 0
            for _ in range(2):
                eval_env = CardinalV2Aviary(
                    act=DEFAULT_ACT,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    fixed_direction=direction,
                )
                
                obs, _ = eval_env.reset()
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                
                if info.get("collision_occurred", False):
                    hits += 1
                    total_collisions += 1
                
                eval_env.close()
            
            status = "✓✓" if hits == 2 else ("✓" if hits == 1 else "✗")
            print(f"  {direction.upper():>8}: {hits}/2 [{status}]")
        
        print(f"  TOTAL: {total_collisions}/8")
        return total_collisions


def train_stage(stage_config, model=None, n_envs=5, max_timesteps=5_000_000):
    """Train a single curriculum stage."""
    
    name = stage_config["name"]
    env_class = stage_config["env_class"]
    n_eval = stage_config["n_eval"]
    description = stage_config["description"]
    is_cardinal = stage_config.get("is_cardinal", False)
    
    print(f"\n{'#'*70}")
    print(f"# CURRICULUM STAGE: {name.upper()}")
    print(f"# {description}")
    print(f"# Eval: {n_eval} tests | Max timesteps: {max_timesteps:,}")
    print(f"{'#'*70}\n")
    
    # Create env factory
    def make_env(rank):
        def _init():
            env = env_class(
                act=DEFAULT_ACT,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
            )
            return Monitor(env)
        return _init
    
    # Create parallel training environments
    sb3_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create or update model
    if model is None:
        print("[INFO] Creating NEW PPO model (Stage 1)")
        model = PPO(
            "MlpPolicy",
            sb3_env,
            verbose=1,
            tensorboard_log="./ppo_curriculum_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    else:
        print("[INFO] Continuing from previous stage model")
        model.set_env(sb3_env)
    
    # Eval callback
    eval_callback = CurriculumEvalCallback(
        env_class=env_class,
        n_eval=n_eval,
        eval_freq=50000,
        is_cardinal=is_cardinal,
        verbose=1,
    )
    
    # Train
    model.learn(
        total_timesteps=max_timesteps,
        callback=eval_callback,
        tb_log_name=f"curriculum_{name}",
        reset_num_timesteps=False,  # Continue timestep count
    )
    
    # Save stage model
    model_path = f"models/curriculum_part_{name}"
    model.save(model_path)
    print(f"[INFO] Stage '{name}' model saved to {model_path}")
    
    sb3_env.close()
    
    return model, eval_callback.perfect_achieved


def run_curriculum(n_envs=5, max_timesteps_per_stage=5_000_000):
    """Run the full curriculum learning pipeline."""
    
    print("="*70)
    print("CURRICULUM LEARNING - DRONE TARGET INTERCEPTION")
    print("="*70)
    print(f"Stages: {len(STAGES)}")
    for i, stage in enumerate(STAGES):
        print(f"  {i+1}. {stage['name'].upper()}: {stage['description']}")
    print(f"Max timesteps per stage: {max_timesteps_per_stage:,}")
    print(f"Parallel envs: {n_envs}")
    print("="*70)
    
    model = None
    results = []
    
    for i, stage in enumerate(STAGES):
        print(f"\n>>> Starting Stage {i+1}/{len(STAGES)}: {stage['name'].upper()}")
        
        model, perfect = train_stage(
            stage,
            model=model,
            n_envs=n_envs,
            max_timesteps=max_timesteps_per_stage,
        )
        
        results.append({
            "stage": stage["name"],
            "perfect": perfect,
        })
        
        if not perfect:
            print(f"\n[WARNING] Stage '{stage['name']}' did not achieve perfect score!")
            print(f"[INFO] Continuing to next stage anyway...")
    
    # Final summary
    print("\n" + "="*70)
    print("CURRICULUM COMPLETE!")
    print("="*70)
    for r in results:
        status = "✓ PERFECT" if r["perfect"] else "✗ Incomplete"
        print(f"  {r['stage'].upper():>12}: {status}")
    
    # Save final model
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    final_path = f"models/curriculum_final_{timestamp}"
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Curriculum Learning for Drone Target Interception")
    parser.add_argument("--n_envs", type=int, default=5, help="Number of parallel environments")
    parser.add_argument("--max_timesteps", type=int, default=5_000_000, help="Max timesteps per stage")
    args = parser.parse_args()
    
    run_curriculum(n_envs=args.n_envs, max_timesteps_per_stage=args.max_timesteps)
