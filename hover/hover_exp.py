import cv2
import numpy as np
import torch as th

import moviepy.editor as mp
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import pybullet as p

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Video
from stable_baselines3 import PPO

from TunedHoverAviary import TunedHoverAviary

CAM_LOOKAT_POS = [0,1,0.5]
CAM_POS = [1,1,1.5]

def get_frame(cid, target_pos, cam_pos):
    # define your camera pose
    view = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=[0, 0, 1],
        physicsClientId=cid,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=80.0,
        aspect=1.0,
        nearVal=0.05,
        farVal=10.0,
        physicsClientId=cid,
    )

    w, h = 720, 720
    _, _, rgb, depth, seg = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        physicsClientId=cid,
        renderer=p.ER_BULLET_HARDWARE_OPENGL, # or leave default
    )

    img = np.reshape(rgb, (h, w, 4))  # BGRA
    # bgr = img[...,:3]  
    # cv2.imwrite("frame.png", bgr)
    return img

class CustomEvalCallback(BaseCallback):
    """
    Evaluates the agent periodically and logs mean reward, std, and optionally frames to TensorBoard.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, log_name="eval", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_name = log_name

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = self.evaluate_and_log()
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        return True

    def evaluate_and_log(self):
        rewards = []
        frames = []
        cid = self.eval_env.getPyBulletClient()

        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_reward = 0
            frames = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward

                # Capture frame for TensorBoard
                if self.logger:
                    frame = get_frame(cid, CAM_LOOKAT_POS, CAM_POS)
                    # frame = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
                    frame = np.transpose(frame, (2,0,1))  # HWC -> CHW
                    frames.append(frame)
                    # try:
                    #     # frame = self.eval_env.render(mode="rgb_array")
                    #     frame = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
                    #     frame = np.transpose(frame, (2,0,1))  # HWC -> CHW
                    #     self.logger.get_writer().add_image(f"{self.log_name}/frame_episode{ep}", frame, self.num_timesteps)
                    # except Exception as e:
                    #     if self.verbose > 0:
                    #         print(f"Could not render frame: {e}")

            rewards.append(ep_reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Log scalar metrics
        if self.logger:
            self.logger.record(f"{self.log_name}/mean_reward", mean_reward)
            self.logger.record(f"{self.log_name}/std_reward", std_reward)
            self.logger.dump(self.num_timesteps)
            self.logger.record(
                "trajectory/video",
                Video(th.from_numpy(np.asarray([frames])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
            print("logged video")
        return mean_reward, std_reward



# We will use the 1D-RPM to control the hover
# Observation type will be limited to Kinematic information (no RGB)
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


INIT_XYZS = np.array([[0, 0, 1]])
INIT_RPYS = np.array([[0, 0, 0]])


def run(from_model=None):
    train_env = TunedHoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    # eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    PYB_CLIENT = train_env.getPyBulletClient()

    sb3_env = DummyVecEnv([lambda: Monitor(train_env)])
    
    model = None
    if from_model:
        try:
            model = PPO.load(from_model, env=sb3_env, verbose=1,
                tensorboard_log="./ppo_tensorboard/")
            print("loaded model")
        except Exception as e:
            print("new model")
            model = PPO("MlpPolicy", sb3_env, verbose=1, 
                tensorboard_log="./ppo_tensorboard/")
    else:
        print("new model")
        model = PPO("MlpPolicy", sb3_env, verbose=1, 
            tensorboard_log="./ppo_tensorboard/")

    eval_callback = CustomEvalCallback(train_env, eval_freq=10000, n_eval_episodes=1)

    model.learn(total_timesteps=300000, callback=eval_callback, tb_log_name="PPO")
    print("saving model.")
    model.save("models/ppo_hover_model_4d_600k_updated")
    return

if __name__ == "__main__":
    run(from_model="models/ppo_hover_model_4d_600k_updated.zip")