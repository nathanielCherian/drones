"""
True Random Training with Adaptive Velocity Curriculum
=======================================================
Trains on TrueRandomAviary with progressively increasing velocity.

Velocity Curriculum:
- Start: [2, 4] cm/s
- Each time 10/10 eval achieved, increase range by 0.5 cm/s
- Once max_speed > 6 cm/s, start saving models (max_velo_6, max_velo_6.5, etc.)
- Stop at max_speed = 400 cm/s (4 m/s)

Eval: 10 preset speeds from min to max, random directions, random positions
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

from TrueRandomAviary import TrueRandomAviary
from gym_pybullet_drones.utils.enums import ActionType

DEFAULT_ACT = ActionType('rpm')
INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])

INITIAL_MIN_SPEED = 0.02
INITIAL_MAX_SPEED = 0.04
SPEED_INCREMENT = 0.005
SAVE_THRESHOLD = 0.06
FINAL_MAX_SPEED = 1.0
N_EVAL = 10
MAX_TIMESTEPS_PER_STAGE = 10_000_000

INITIAL_MAX_XY = 0.5
INITIAL_MAX_Z = 0.8
BOUNDS_INCREMENT = 0.1
MAX_XY_CAP = 7.0
MAX_Z_CAP = 3.0


class AdaptiveVelocityCallback(BaseCallback):
    
    def __init__(self, min_speed, max_speed, max_xy, max_z, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_xy = max_xy
        self.max_z = max_z
        self.eval_freq = eval_freq
        self.last_eval_timestep = 0
        self.perfect_achieved = False
        self.best_hits = 0
        
    def _get_eval_speeds(self):
        return np.linspace(self.min_speed, self.max_speed, N_EVAL)
    
    def _on_step(self):
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            
            speeds = self._get_eval_speeds()
            print(f"\n{'='*60}")
            print(f"[EVAL @ {self.num_timesteps}] Testing {N_EVAL} speeds")
            print(f"  Speed range: {self.min_speed*100:.1f} - {self.max_speed*100:.1f} cm/s")
            print(f"  Bounds: XY=±{self.max_xy:.2f} Z=[0.4, {self.max_z:.2f}]")
            print(f"{'='*60}")
            
            hits = 0
            for i, speed in enumerate(speeds):
                angle = np.random.uniform(0, 2 * np.pi)
                direction = [np.cos(angle), np.sin(angle), 0.0]
                
                eval_env = TrueRandomAviary(
                    act=DEFAULT_ACT,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    fixed_speed=speed,
                    fixed_direction=direction,
                    max_xy=self.max_xy,
                    max_z=self.max_z,
                )
                
                obs, _ = eval_env.reset()
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                
                hit = info.get("collision_occurred", False)
                if hit:
                    hits += 1
                
                status = "✓" if hit else "✗"
                print(f"  Speed {speed*100:5.1f} cm/s: {status}")
                
                eval_env.close()
            
            print(f"\n  TOTAL: {hits}/{N_EVAL}")
            
            if hits > self.best_hits:
                self.best_hits = hits
                print(f"  -> New best! ({hits}/{N_EVAL})")
            
            if hits == N_EVAL:
                print(f"\n  *** PERFECT {N_EVAL}/{N_EVAL}! Ready to advance! ***")
                self.perfect_achieved = True
                return False
            
            print(f"{'='*60}\n")
        
        return True


def train_velocity_curriculum(from_model, n_envs=9, total_timesteps_limit=10_000_000):
    print("="*70)
    print("TRUE RANDOM - ADAPTIVE VELOCITY CURRICULUM")
    print("="*70)
    print(f"Starting from: {from_model}")
    print(f"Initial speed range: {INITIAL_MIN_SPEED*100:.1f} - {INITIAL_MAX_SPEED*100:.1f} cm/s")
    print(f"Speed increment: {SPEED_INCREMENT*100:.1f} cm/s per stage")
    print(f"Bounds increment: {BOUNDS_INCREMENT*100:.1f} cm per stage")
    print(f"Max XY cap: {MAX_XY_CAP:.1f} m")
    print(f"Max Z cap: {MAX_Z_CAP:.1f} m")
    print(f"Save models when max > {SAVE_THRESHOLD*100:.1f} cm/s")
    print(f"Final max speed: {FINAL_MAX_SPEED*100:.1f} cm/s")
    print("="*70)
    
    min_speed = INITIAL_MIN_SPEED
    max_speed = INITIAL_MAX_SPEED
    max_xy = INITIAL_MAX_XY
    max_z = INITIAL_MAX_Z
    
    def make_env(rank):
        def _init():
            env = TrueRandomAviary(
                act=DEFAULT_ACT,
                initial_xyzs=INIT_XYZS,
                initial_rpys=INIT_RPYS,
                min_speed=INITIAL_MIN_SPEED,
                max_speed=INITIAL_MAX_SPEED,
                max_xy=INITIAL_MAX_XY,
                max_z=INITIAL_MAX_Z,
            )
            return Monitor(env)
        return _init
    
    sb3_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    model = PPO.load(from_model, env=sb3_env)
    print(f"[INFO] Loaded base model from {from_model}")
    print(f"[INFO] Created {n_envs} parallel environments")
    
    initial_num_timesteps = model.num_timesteps
    print(f"[INFO] Model has {initial_num_timesteps} timesteps from previous training")
    
    stage = 0
    while max_speed <= FINAL_MAX_SPEED and (model.num_timesteps - initial_num_timesteps) < total_timesteps_limit:
        stage += 1
        print(f"\n{'#'*70}")
        print(f"# STAGE {stage}: Speed [{min_speed*100:.1f}, {max_speed*100:.1f}] cm/s | Bounds XY:±{max_xy:.2f} Z:[0.4,{max_z:.2f}]")
        print(f"{'#'*70}\n")

        sb3_env.env_method("set_speed_range", min_speed, max_speed, max_xy, max_z)

        callback = AdaptiveVelocityCallback(
            min_speed=min_speed,
            max_speed=max_speed,
            max_xy=max_xy,
            max_z=max_z,
            eval_freq=50000,
        )

        model.learn(
            total_timesteps=MAX_TIMESTEPS_PER_STAGE,
            callback=callback,
            tb_log_name=f"true_random_v{max_speed*100:.0f}",
            reset_num_timesteps=False,
        )

        if max_speed >= SAVE_THRESHOLD:
            model_name = f"models/max_velo_vert_{int(max_speed*100)}"
            model.save(model_name)
            print(f"[INFO] Saved model: {model_name}")

        if callback.perfect_achieved:
            print(f"[INFO] Stage {stage} complete! Advancing velocity...")
        else:
            print(f"[WARNING] Stage {stage} did not achieve 10/10, advancing anyway...")

        old_min, old_max = min_speed, max_speed
        old_xy, old_z = max_xy, max_z
        
        min_speed += SPEED_INCREMENT
        max_speed += SPEED_INCREMENT
        max_xy = min(max_xy + BOUNDS_INCREMENT, MAX_XY_CAP)
        max_z = min(max_z + BOUNDS_INCREMENT, MAX_Z_CAP)
        
        print(f"\n{'*'*60}")
        print(f"*** VELOCITY & BOUNDS INCREASED ***")
        print(f"  Speed Floor: {old_min*100:.1f} cm/s -> {min_speed*100:.1f} cm/s")
        print(f"  Speed Ceiling: {old_max*100:.1f} cm/s -> {max_speed*100:.1f} cm/s")
        print(f"  Bounds XY: ±{old_xy:.2f} -> ±{max_xy:.2f}")
        print(f"  Bounds Z: [0.4, {old_z:.2f}] -> [0.4, {max_z:.2f}]")
        print(f"{'*'*60}\n")
    
    sb3_env.close()
    
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    final_path = f"models/true_random_final_{timestamp}"
    model.save(final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Final max speed: {(max_speed - SPEED_INCREMENT)*100:.1f} cm/s")
    print(f"Final model: {final_path}")
    print("="*70)
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_model", type=str, 
                        default="models/curriculum_final_12.05.2025_21.57.16",
                        help="Base model to start from")
    parser.add_argument("--n_envs", type=int, default=9,
                        help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000,
                        help="Total timesteps limit for training")
    args = parser.parse_args()
    
    train_velocity_curriculum(from_model=args.from_model, n_envs=args.n_envs, total_timesteps_limit=args.total_timesteps)