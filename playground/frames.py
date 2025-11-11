import os
import time
import argparse
from datetime import datetime
import pdb
import cv2
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def render_from_custom_cam(cid, target_pos, cam_pos):

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
    bgr = img[...,:3]  
    cv2.imwrite("frame.png", bgr)
    return img


def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(1)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/1] for i in range(1)])

    print("INIT XYS: ", INIT_XYZS)
    print("INIT RPYS: ", INIT_RPYS)

    env = CtrlAviary(drone_model=drone,
                        num_drones=1,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    PYB_CLIENT = env.getPyBulletClient()

    controller = DSLPIDControl(drone_model=drone)


    action = np.zeros((1,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        action[0, :], _, _ = controller.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                state=obs[0],
                                target_pos=INIT_XYZS[0],
                                # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                target_rpy=INIT_RPYS[0]
                            )

        # action[0, :] = np.array([14447.05621394, 14447.05621394, 14447.05621394, 14447.05621394])

        env.render()

        if i == 0:
            render_from_custom_cam(PYB_CLIENT, INIT_XYZS[0], [0.5,0.5,0.4])
            
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

            keys = p.getKeyboardEvents()
            pressed_char = lambda c: ord(c) in keys and (keys[ord(c)] & p.KEY_IS_DOWN)
            if pressed_char('l'):
                INIT_XYZS[0,0] += 0.05
            if pressed_char('j'):
                INIT_XYZS[0,0] -= 0.05
            if pressed_char('i'):
                INIT_XYZS[0,1] += 0.05
            if pressed_char('j'):
                INIT_XYZS[0,1] -= 0.05
            if pressed_char('m'):
                INIT_XYZS[0,2] += 0.05
            if pressed_char('n'):
                INIT_XYZS[0,2] -= 0.05

    env.close()

if __name__ == "__main__":
    run()