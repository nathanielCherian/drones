import cv2
import numpy as np
import torch as th

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import pybullet as p

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Video, Image
from stable_baselines3 import PPO

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image as PILImage

from NoisyHoverAviary import NoisyHoverAviary

CAM_LOOKAT_POS = [0,0,0.5]
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

def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    pil_img = PILImage.open(buf).convert("RGB")
    img_array = np.array(pil_img)

    buf.close()
    plt.close(fig)
    return img_array

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
            values = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward

                state = self.eval_env._getDroneStateVector(0)

                # Capture frame for TensorBoard
                if self.logger:
                    # if len(frames) < self.eval_env.PYB_FREQ*2:
                    #     frame = get_frame(cid, CAM_LOOKAT_POS, CAM_POS)
                    #     # frame = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
                    #     frame = np.transpose(frame, (2,0,1))  # HWC -> CHW
                    #     frames.append(frame)

                    TARGET_POS = np.array([0,0,1])
                    norm = np.linalg.norm(TARGET_POS-state[0:3])
                    values.append(norm)

            rewards.append(ep_reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        fig, ax = plt.subplots()
        ax.plot([i/self.eval_env.PYB_FREQ for i in range(len(values))], values)
        ax.set_title("Euclidian Distance")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        # Log scalar metrics
        if self.logger:
            self.logger.record(f"{self.log_name}/mean_reward", mean_reward)
            self.logger.record(f"{self.log_name}/std_reward", std_reward)
            self.logger.dump(self.num_timesteps)
            # self.logger.record(
            #     "trajectory/video",
            #     Video(th.from_numpy(np.asarray([frames])), fps=40),
            #     exclude=("stdout", "log", "json", "csv"),
            # )
            print("logged video")


            img = fig_to_image(fig)
            self.logger.record(
                f"trajectory/euclidian",
                Image(img, dataformats='HWC'),
                exclude=("stdout", "log", "json", "csv"),
            )
            print("logged euclidian distance")

        return mean_reward, std_reward



# We will use the 1D-RPM to control the hover
# Observation type will be limited to Kinematic information (no RGB)
DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


INIT_XYZS = np.array([[0, 0, 0]])
INIT_RPYS = np.array([[0, 0, 0]])

def run(from_model=None, save_model=None):
    train_env = NoisyHoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS, noise=0.25)
    # eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT_XYZS, initial_rpys=INIT_RPYS)
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    PYB_CLIENT = train_env.getPyBulletClient()

    sb3_env = DummyVecEnv([lambda: Monitor(train_env)])
    
    model = None
    if from_model:
        model = PPO.load(from_model, env=sb3_env, verbose=1,
            tensorboard_log="./final_logs/")
    else:
        model = PPO("MlpPolicy", sb3_env, verbose=1, 
            tensorboard_log="./final_logs/", )

    eval_callback = CustomEvalCallback(train_env, eval_freq=10000, n_eval_episodes=1)

    model.learn(total_timesteps=200000, callback=eval_callback, tb_log_name=save_model)
    if save_model:
        print("saving model.")
        model.save(save_model)


"""
My record of runs
# Changing the exponent
1. Reward function max(0, 4 - (norm)**1), length=15 seconds, epochs=200k
2. Reward function max(0, 4 - (norm)**2), length=15 seconds, epochs=200k
3. Reward function max(0, 4 - (norm)**3), length=15 seconds, epochs=200k

# Adding extra boost
4. Reward function max(0, 4 - (norm)**2) (0.5 radius + 2), length=15 seconds, epochs=200k

# Gaussian reward function
5. Reward function gaussian, p=2, k=1 length=15 seconds, epochs=200k

6. Reward function gaussian, p=2, k=2 length=15 seconds, epochs=200k
    - y error is doubled

7. Reward function gaussian, p=2, k=2 length=15 seconds, epochs=200k
    - Low ceiling

next up...
7. Reward function gaussian, p=1, k=1 length=15 seconds, epochs=200k
8. Reward function gaussian, p=2, k=0.5 length=15 seconds, epochs=200k
9. Reward function gaussian, p=2, k=2 length=15 seconds, epochs=200k


10. Reward function gaussian, p=2, k=1, jitter + rotation penalties length=15 seconds, epochs=200k
    - Jitter penalty = norm of velocity
    - rotation penality = norm of rotation velocity
"""


if __name__ == "__main__":
    run(save_model="noisyact_PPO_200k_15s_r2")