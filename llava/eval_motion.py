from pathlib import Path

from matplotlib import pyplot as plt
from unrealcv import Client
from typing import Union, List, Dict, Any
import cv2
import io
import time
import math
import subprocess, threading
import airsim
from common import *
import psutil
import requests
import random
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
import os, json
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# from data_test import model_path
# from extern.hf.configuration_prismatic import OpenFlyConfig
# from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
# from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# from model.load_model import OpenFly
# from model.vision_backbone import DinoSigLIPViTBackbone
# from model.llm_backbone import LLaMa2LLMBackbone
from llava.model import *
from llava.model.action_tokenizer import ActionTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# AutoConfig.register("openvla", OpenFlyConfig)
# AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
# AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
# AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])


class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def _init_airsim_sim(self):
        env_dir = "../envs/airsim/" + self.env_name
        # env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type='color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")


class UEBridge:
    def __init__(self, ue_ip, ue_port, env_name):
        self.kill_failed_process()
        time.sleep(10)

        # port = self.find_available_port()

        port = random.randint(9000, 9100)
        print(f"Available port: {port}")
        self.modify_port_in_ini(port, env_name)
        ue_port = port

        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_ue_sim)
        self._sim_thread.start()
        time.sleep(15)

        self._client = Client((ue_ip, ue_port))
        self._connection_check()

        self._camera_init()

        # self._drone_init()
        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def find_available_port(self):
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()

            if f'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        # ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        ini_file = f"../envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"

        with open(ini_file, 'r') as file:
            lines = file.readlines()

        with open(ini_file, 'w') as file:
            for line in lines:
                if line.startswith("Port="):
                    file.write(f"Port={port}\n")
                else:
                    file.write(line)

    def kill_failed_process(self):
        result = subprocess.run(['pgrep', '-n', 'CrashReport'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

        result = subprocess.run(['pgrep', '-n', 'CitySample'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

    def _init_ue_sim(self):
        env_dir = "../envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/CitySample.sh"]

        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)
        time.sleep(2)

    def __del__(self):
        self._client.disconnect()

    def _connection_check(self):
        '''Check if connected'''
        if self._client.connect():
            print('UnrealCV connected successfully')
        else:
            print('UnrealCV is not connected')
            exit()

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        '''Set camera position'''
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        print('camera_settings', camera_settings)

    def _camera_init(self):
        '''Camera initialization'''
        time.sleep(2)
        self._client.request('vset /cameras/spawn')
        self._client.request('vset /camera/1/size 1920 1080')
        time.sleep(2)
        self.set_camera_pose(150, 400, 15, 0, 0, 0)  # Initial position
        time.sleep(2)

    def get_camera_data(self, camera_type='lit'):
        valid_types = {'lit', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'lit':
            data = self._client.request('vget /camera/1/lit png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'object_mask':
            data = self._client.request('vget /camera/1/object_mask png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'depth':
            data = self._client.request('vget /camera/1/depth npy')
            depth_np = np.load(io.BytesIO(data))
            return depth_np  # Return depth data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='lit'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)


class GSBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def _init_gs_sim(self):
        # dataset_dir = "envs/gs/" + self.env_name
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw

    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text)
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass


def get_images(lst, if_his, step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0], lst[0], lst[0]]


def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result


def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):
    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)

    prompt = text
    inputs = processor(prompt, images).to("cuda:2", dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action


def get_action2(policy, image_list, text, his, if_his=False, his_step=0):
    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)

    prompt = text
    # inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
    action = policy.predict_action(images, prompt, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action


def get_action3(model, image_list, text, norm_stats, action_tokenizer, same_seq=False, return_seq=False):
    """如果 return_seq=True，则返回 (action_id, seq_last50) 其中 seq_last50 是 (<=50, D) 的数组"""
    image = image_list[-1]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    captured_seq = None

    def hook_fn(module, input, output):
        nonlocal captured_seq
        if output is not None and isinstance(output, torch.Tensor):
            if output.dim() == 3:
                seq = output[0].detach().cpu().numpy()  # (L, D)
            elif output.dim() == 2:
                seq = output.detach().cpu().numpy()  # (L, D)
            else:
                seq = output.detach().cpu().numpy().flatten()
            captured_seq = seq

    handle = model.history_mamba.register_forward_hook(hook_fn)
    action = model.predict_action(image, text, norm_stats=norm_stats,
                                  action_tokenizer=action_tokenizer,
                                  unnorm_key="vlnv1", same_seq=same_seq)
    action = action.round().astype(int)
    action_id = convert_to_action_id(action)

    print("Action:", action_id)
    handle.remove()

    if return_seq and captured_seq is not None and captured_seq.ndim == 2:
        last_n = 50
        if len(captured_seq) >= last_n:
            last50 = captured_seq[-last_n:, :]
        else:
            last50 = captured_seq
        return action_id, last50
    else:
        return action_id, None


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 +
                     (point2[1] - point1[1]) ** 2 +
                     (point2[2] - point1[2]) ** 2)


def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) * 2
        y += step_size * math.sin(yaw) * 2
    elif action == 9:
        x += step_size * math.cos(yaw) * 3
        y += step_size * math.sin(yaw) * 3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]


def flatten_state_dict(model_state_dicts):
    """
    把 {"vision_backbone": {"layer.weight": ...}, "llm_backbone": {...}}
    转换为 {"vision_backbone.layer.weight": ..., "llm_backbone.xxx": ...}
    """
    flat = {}
    for mkey, subdict in model_state_dicts.items():
        for subkey, tensor in subdict.items():
            flat[f"{mkey}.{subkey}"] = tensor
    return flat


# ==================== 新增：运动状态捕获证明函数 ====================
# ==================== 修复后的 prove_motion_capture 函数 ====================
def prove_motion_capture(states, actions, positions, sample_dir, env_name, idx):
    states = np.vstack(states)
    actions = np.array(actions)
    positions = np.array(positions)
    T = len(states)

    if T < 5:
        print(f"Sample {idx}: too few steps ({T}), skip motion proof.")
        return None

    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. PCA 降维（安全初始化）----------
    n_comp = min(10, states.shape[0], states.shape[1])
    pca = PCA(n_components=n_comp)
    states_pca = pca.fit_transform(states)
    use_dim = min(5, n_comp)

    # ---------- 2. 动作解码实验（修复 CV 问题）----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(states_pca[:, :use_dim])

    n_classes = len(np.unique(actions))

    # 修复：检查每个类别的样本数，避免 CV fold 中只有一个类别
    from collections import Counter
    action_counts = Counter(actions)
    min_count = min(action_counts.values())

    # 安全设置 CV folds：不能超过最小类别样本数
    safe_cv = min(5, min_count, T // 2) if min_count >= 2 else 2
    if safe_cv < 2:
        safe_cv = 2

    # 如果类别数太少或样本太少，跳过 CV，直接全量训练评估
    if n_classes < 2 or T < 8 or min_count < 2:
        # 直接全量训练+预测（无 CV）
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
        clf.fit(X_scaled, actions)
        pred_actions = clf.predict(X_scaled)
        action_acc = accuracy_score(actions, pred_actions)
    else:
        try:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
            action_scores = cross_val_score(clf, X_scaled, actions, cv=skf)
            action_acc = action_scores.mean()
        except Exception as e:
            # CV 失败回退到全量评估
            clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
            clf.fit(X_scaled, actions)
            pred_actions = clf.predict(X_scaled)
            action_acc = accuracy_score(actions, pred_actions)

    # 全量训练用于混淆矩阵
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
    clf.fit(X_scaled, actions)
    pred_actions = clf.predict(X_scaled)
    action_cm = np.zeros((10, 10), dtype=int)
    for a, p in zip(actions, pred_actions):
        action_cm[a, p] += 1

    # ---------- 3. 位置/航向回归 ----------
    reg_xyz = Ridge(alpha=1.0)
    reg_xyz.fit(states_pca[:, :use_dim], positions[:, :3])
    pos_pred = reg_xyz.predict(states_pca[:, :use_dim])
    pos_r2_xyz = r2_score(positions[:, :3], pos_pred)
    pos_r2_x = r2_score(positions[:, 0], pos_pred[:, 0])
    pos_r2_y = r2_score(positions[:, 1], pos_pred[:, 1])
    pos_r2_z = r2_score(positions[:, 2], pos_pred[:, 2])

    reg_yaw = Ridge(alpha=1.0)
    reg_yaw.fit(states_pca[:, :use_dim], positions[:, 3])
    yaw_pred = reg_yaw.predict(states_pca[:, :use_dim])
    yaw_r2 = r2_score(positions[:, 3], yaw_pred)

    # ---------- 4. 速度/位移相关性 ----------
    if T >= 3:
        phys_disp = np.diff(positions[:, :3], axis=0)
        phys_speed = np.linalg.norm(phys_disp, axis=1)

        vel_dim = min(3, n_comp)
        state_diff = np.diff(states_pca[:, :vel_dim], axis=0)
        state_speed = np.linalg.norm(state_diff, axis=1)

        if np.std(state_speed) > 1e-6 and np.std(phys_speed) > 1e-6:
            corr_speed = np.corrcoef(state_speed, phys_speed)[0, 1]
        else:
            corr_speed = 0.0

        phys_disp_norm = phys_disp / (np.linalg.norm(phys_disp, axis=1, keepdims=True) + 1e-8)
        state_diff_norm = state_diff / (np.linalg.norm(state_diff, axis=1, keepdims=True) + 1e-8)
        direction_alignment = np.sum(phys_disp_norm[:, :vel_dim] * state_diff_norm, axis=1)
        mean_alignment = np.mean(direction_alignment)

        corr_dim = []
        for i in range(min(3, vel_dim)):
            if np.std(state_diff[:, i]) > 1e-6 and np.std(phys_disp[:, i]) > 1e-6:
                c = np.corrcoef(state_diff[:, i], phys_disp[:, i])[0, 1]
                corr_dim.append(c)
            else:
                corr_dim.append(0.0)
    else:
        corr_speed = 0.0
        mean_alignment = 0.0
        corr_dim = [0.0, 0.0, 0.0]

    # ---------- 5. 可视化（修复 tight_layout 溢出）----------
    time_steps = np.arange(T)
    action_names = {0: 'Stop', 1: 'Fwd', 2: 'L30', 3: 'R30', 4: 'Up',
                    5: 'Dn', 6: 'Lf', 7: 'Rf', 8: 'F2', 9: 'F3'}

    # 5.1 3D 状态演化图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(T - 1):
        ax.plot(states_pca[i:i + 2, 0], states_pca[i:i + 2, 1], time_steps[i:i + 2],
                'k-', alpha=0.3, linewidth=1)
    sc = ax.scatter(states_pca[:, 0], states_pca[:, 1], time_steps,
                    c=actions, cmap='tab10', s=80, alpha=0.9,
                    edgecolors='k', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel('Decision Step')
    ax.set_title(f'Mamba State Evolution (Action-colored)\n{env_name} sample {idx}', fontsize=12)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Action ID')
    plt.savefig(sample_dir / 'state_evolution_action_colored.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5.2 物理轨迹 3D 图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k-', alpha=0.3, linewidth=1)
    sc2 = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     c=time_steps, cmap='plasma', s=80, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Physical Flight Trajectory\n{env_name} sample {idx}', fontsize=12)
    cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.5)
    cbar2.set_label('Decision Step')
    plt.savefig(sample_dir / 'physical_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5.3 状态-动作分离图
    plt.figure(figsize=(9, 7))
    for act_id in np.unique(actions):
        mask = actions == act_id
        plt.scatter(states_pca[mask, 0], states_pca[mask, 1],
                    label=action_names.get(act_id, f'A{act_id}'),
                    alpha=0.8, s=100, edgecolors='k', linewidth=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'State-Action Separability | Acc: {action_acc:.3f} (chance: {1 / n_classes:.3f})', fontsize=11)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(sample_dir / 'state_action_separation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5.4 运动相关性综合分析图（修复 tight_layout 溢出）
    if T >= 3:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # (a) 状态速度 vs 物理速度
        axes[0, 0].scatter(state_speed, phys_speed, alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        if len(state_speed) > 1 and np.std(state_speed) > 1e-6:
            z = np.polyfit(state_speed, phys_speed, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(state_speed, p(state_speed), "r--", alpha=0.8, linewidth=2)
        axes[0, 0].set_xlabel('State Change Speed')
        axes[0, 0].set_ylabel('Physical Speed (m/step)')
        axes[0, 0].set_title(f'(a) Speed Correlation r={corr_speed:.3f}')
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)

        # (b) 方向一致性
        axes[0, 1].hist(direction_alignment, bins=min(15, len(direction_alignment)),
                        color='steelblue', edgecolor='k', alpha=0.7)
        axes[0, 1].axvline(mean_alignment, color='r', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_alignment:.3f}')
        axes[0, 1].set_xlabel('Direction Alignment (cos)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('(b) Direction Alignment')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)

        # (c) 各维度位移相关性
        dims = ['X', 'Y', 'Z']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        bars = axes[1, 0].bar(dims[:len(corr_dim)], corr_dim, color=colors[:len(corr_dim)],
                              edgecolor='k', alpha=0.85)
        axes[1, 0].set_ylabel('Pearson r')
        axes[1, 0].set_title('(c) Per-Dimension Displacement Corr')
        axes[1, 0].set_ylim(-1, 1)
        axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)
        for bar, val in zip(bars, corr_dim):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                            height + 0.05 * np.sign(height) + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # (d) 位置/航向回归 R²
        pos_metrics = [pos_r2_x, pos_r2_y, pos_r2_z, yaw_r2]
        pos_labels = ['X', 'Y', 'Z', 'Yaw']
        colors2 = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        bars2 = axes[1, 1].bar(pos_labels, pos_metrics, color=colors2, edgecolor='k', alpha=0.85)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('(d) Position/Heading Decoding (Ridge)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)
        for bar, val in zip(bars2, pos_metrics):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 修复：不用 tight_layout，手动调整
        plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3)
        fig.suptitle(f'Motion Capture Proof | {env_name} sample {idx}', fontsize=13, y=0.98)
        plt.savefig(sample_dir / 'motion_correlation_analysis.png', dpi=150)
        plt.close()

    # 5.5 动作解码混淆矩阵
    plt.figure(figsize=(8, 7))
    plt.imshow(action_cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(shrink=0.8)
    plt.xlabel('Predicted Action')
    plt.ylabel('True Action')
    plt.title(f'Action Decoding Confusion Matrix\nAccuracy: {action_acc:.3f}', fontsize=11)
    for i in range(10):
        for j in range(10):
            if action_cm[i, j] > 0:
                plt.text(j, i, str(action_cm[i, j]), ha='center', va='center',
                         color='white' if action_cm[i, j] > action_cm.max() / 2 else 'red',
                         fontsize=9, fontweight='bold')
    plt.savefig(sample_dir / 'action_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- 6. 保存报告 ----------
    report = {
        'sample': int(idx),
        'env': env_name,
        'trajectory_length': int(T),
        'action_decode_accuracy': float(action_acc),
        'action_decode_chance': float(1.0 / n_classes),
        'position_regression_r2_xyz': float(pos_r2_xyz),
        'position_regression_r2_x': float(pos_r2_x),
        'position_regression_r2_y': float(pos_r2_y),
        'position_regression_r2_z': float(pos_r2_z),
        'yaw_regression_r2': float(yaw_r2),
        'state_speed_vs_phys_speed_corr': float(corr_speed),
        'direction_alignment_mean': float(mean_alignment),
        'displacement_corr_x': float(corr_dim[0]),
        'displacement_corr_y': float(corr_dim[1]),
        'displacement_corr_z': float(corr_dim[2]),
        'pca_explained_variance_ratio_top3': [float(x) for x in pca.explained_variance_ratio_[:3].tolist()],
    }

    with open(sample_dir / 'motion_proof_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  MOTION PROOF | {env_name} Sample {idx}")
    print(f"{'=' * 60}")
    print(f"  Trajectory Length      : {T} steps")
    print(f"  Action Decode Accuracy : {action_acc:.4f}  (chance: {1 / n_classes:.4f})")
    print(f"  Position R² (X/Y/Z)    : {pos_r2_x:.3f} / {pos_r2_y:.3f} / {pos_r2_z:.3f}")
    print(f"  Yaw R²                 : {yaw_r2:.4f}")
    print(f"  Speed Correlation      : {corr_speed:.4f}")
    print(f"  Direction Alignment    : {mean_alignment:.4f}")
    print(f"{'=' * 60}\n")

    return report


def plot_2d_time_projection(states, actions, positions, sample_dir, env_name, idx):
    """
    2D 投影可视化：纵轴为时间轴（Decision Step），横轴为状态主成分。
    直观展示 Mamba 状态随时间演化的轨迹。
    """
    states = np.vstack(states)
    actions = np.array(actions)
    positions = np.array(positions)
    T = len(states)

    if T < 3:
        return

    sample_dir = Path(sample_dir)

    # PCA 降维到 2D（安全初始化）
    n_comp = min(2, states.shape[0], states.shape[1])
    if n_comp < 2:
        return
    pca = PCA(n_components=n_comp)
    states_pca = pca.fit_transform(states)

    time_steps = np.arange(T)
    action_names = {0: 'Stop', 1: 'Fwd', 2: 'L30', 3: 'R30', 4: 'Up',
                    5: 'Dn', 6: 'Lf', 7: 'Rf', 8: 'F2', 9: 'F3'}
    action_colors = {0: '#2c3e50', 1: '#e74c3c', 2: '#3498db', 3: '#2ecc71',
                     4: '#f39c12', 5: '#9b59b6', 6: '#1abc9c', 7: '#e67e22',
                     8: '#34495e', 9: '#16a085'}

    # ==================== 图1: PC1 vs Time（纵轴=时间） ====================
    fig, ax = plt.subplots(figsize=(8, 10))

    # 绘制轨迹连线（按时间顺序）
    for i in range(T - 1):
        ax.plot([states_pca[i, 0], states_pca[i + 1, 0]],
                [time_steps[i], time_steps[i + 1]],
                'k-', alpha=0.4, linewidth=1.5, zorder=1)

    # 绘制散点，按动作类型着色
    for act_id in sorted(np.unique(actions)):
        mask = actions == act_id
        if np.sum(mask) == 0:
            continue
        ax.scatter(states_pca[mask, 0], time_steps[mask],
                   c=action_colors.get(act_id, '#95a5a6'),
                   label=action_names.get(act_id, f'A{act_id}'),
                   s=120, alpha=0.9, edgecolors='k', linewidth=1, zorder=3)

    # 标注起点和终点
    ax.scatter(states_pca[0, 0], time_steps[0], c='green', s=200,
               marker='*', edgecolors='k', linewidth=1.5, label='Start', zorder=5)
    ax.scatter(states_pca[-1, 0], time_steps[-1], c='red', s=200,
               marker='X', edgecolors='k', linewidth=1.5, label='End', zorder=5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel('Decision Step (Time)', fontsize=12)
    ax.set_title(f'Mamba State Evolution: PC1 vs Time\n{env_name} sample {idx}', fontsize=13)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, title='Action Type')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.invert_yaxis()  # 时间从上到下递增更符合直觉（可选）

    plt.tight_layout()
    plt.savefig(sample_dir / 'pc1_vs_time_projection.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ==================== 图2: PC2 vs Time（纵轴=时间） ====================
    fig, ax = plt.subplots(figsize=(8, 10))

    for i in range(T - 1):
        ax.plot([states_pca[i, 1], states_pca[i + 1, 1]],
                [time_steps[i], time_steps[i + 1]],
                'k-', alpha=0.4, linewidth=1.5, zorder=1)

    for act_id in sorted(np.unique(actions)):
        mask = actions == act_id
        if np.sum(mask) == 0:
            continue
        ax.scatter(states_pca[mask, 1], time_steps[mask],
                   c=action_colors.get(act_id, '#95a5a6'),
                   label=action_names.get(act_id, f'A{act_id}'),
                   s=120, alpha=0.9, edgecolors='k', linewidth=1, zorder=3)

    ax.scatter(states_pca[0, 1], time_steps[0], c='green', s=200,
               marker='*', edgecolors='k', linewidth=1.5, label='Start', zorder=5)
    ax.scatter(states_pca[-1, 1], time_steps[-1], c='red', s=200,
               marker='X', edgecolors='k', linewidth=1.5, label='End', zorder=5)

    ax.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_ylabel('Decision Step (Time)', fontsize=12)
    ax.set_title(f'Mamba State Evolution: PC2 vs Time\n{env_name} sample {idx}', fontsize=13)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, title='Action Type')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(sample_dir / 'pc2_vs_time_projection.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ==================== 图3: 联合 2D 投影（横轴=PC1，纵轴=Time，颜色=PC2） ====================
    fig, ax = plt.subplots(figsize=(10, 10))

    # 归一化 PC2 用于颜色映射
    pc2_norm = (states_pca[:, 1] - states_pca[:, 1].min()) / \
               (states_pca[:, 1].max() - states_pca[:, 1].min() + 1e-8)

    for i in range(T - 1):
        # 连线颜色用平均 PC2
        avg_color = plt.cm.coolwarm((pc2_norm[i] + pc2_norm[i + 1]) / 2)
        ax.plot([states_pca[i, 0], states_pca[i + 1, 0]],
                [time_steps[i], time_steps[i + 1]],
                color=avg_color, alpha=0.6, linewidth=2, zorder=1)

    # 散点：横轴=PC1，纵轴=Time，颜色=PC2，大小=动作重要性（可固定）
    sc = ax.scatter(states_pca[:, 0], time_steps,
                    c=states_pca[:, 1], cmap='coolwarm',
                    s=150, alpha=0.9, edgecolors='k', linewidth=1, zorder=3)

    # 标注动作符号
    for i, act in enumerate(actions):
        ax.annotate(action_names.get(act, str(act))[0],  # 取首字母
                    (states_pca[i, 0], time_steps[i]),
                    textcoords="offset points", xytext=(0, 0),
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.scatter(states_pca[0, 0], time_steps[0], c='green', s=250,
               marker='*', edgecolors='k', linewidth=2, label='Start', zorder=5)
    ax.scatter(states_pca[-1, 0], time_steps[-1], c='red', s=250,
               marker='X', edgecolors='k', linewidth=2, label='End', zorder=5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel('Decision Step (Time)', fontsize=12)
    ax.set_title(f'State Trajectory 2D Projection: PC1 vs Time (color=PC2)\n{env_name} sample {idx}', fontsize=13)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('PC2 Value')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(sample_dir / 'state_2d_time_projection_joint.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[2D Time Projection] Saved PC1/PC2 vs Time plots for sample {idx}")

# ==================== 修复后的 visualize_mamba_attention_and_states 函数 ====================
def visualize_mamba_attention_and_states(states, actions, positions, sample_dir, env_name, idx):
    states = np.vstack(states)
    actions = np.array(actions)
    positions = np.array(positions)
    T = len(states)

    if T < 5:
        return

    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    action_names = {0: 'Stop', 1: 'Fwd', 2: 'L30', 3: 'R30', 4: 'Up',
                    5: 'Dn', 6: 'Lf', 7: 'Rf', 8: 'F2', 9: 'F3'}
    action_colors = {0: '#2c3e50', 1: '#e74c3c', 2: '#3498db', 3: '#2ecc71',
                     4: '#f39c12', 5: '#9b59b6', 6: '#1abc9c', 7: '#e67e22',
                     8: '#34495e', 9: '#16a085'}

    # ---------- 1. t-SNE 动作聚类 ----------
    from sklearn.manifold import TSNE

    perplexity = min(30, T - 1) if T > 5 else 2
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    states_tsne = tsne.fit_transform(states)

    plt.figure(figsize=(12, 10))
    for act_id in sorted(np.unique(actions)):
        mask = actions == act_id
        if np.sum(mask) == 0:
            continue
        plt.scatter(states_tsne[mask, 0], states_tsne[mask, 1],
                    c=action_colors.get(act_id, '#95a5a6'),
                    label=action_names.get(act_id, f'A{act_id}'),
                    s=150, alpha=0.85, edgecolors='k', linewidth=1.2, zorder=3)

        act_indices = np.where(mask)[0]
        if len(act_indices) > 1:
            for i in range(len(act_indices) - 1):
                idx1, idx2 = act_indices[i], act_indices[i + 1]
                if idx2 == idx1 + 1:
                    plt.plot([states_tsne[idx1, 0], states_tsne[idx2, 0]],
                             [states_tsne[idx1, 1], states_tsne[idx2, 1]],
                             color=action_colors.get(act_id, '#95a5a6'),
                             alpha=0.4, linewidth=1.5, zorder=1)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'Implicit Motion Manifold: Mamba States Cluster by Action Type\n'
              f'{env_name} sample {idx} (T={T})')
    plt.legend(loc='best', fontsize=10, framealpha=0.9, title='Motion Type')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(sample_dir / 'qualitative_tsne_action_manifold.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---------- 2. 状态速度场 (Quiver Plot) ----------
    if T >= 3:
        # 修复：安全初始化 PCA
        n_comp_2 = min(2, states.shape[0], states.shape[1])
        pca = PCA(n_components=n_comp_2)
        states_pca = pca.fit_transform(states)

        state_vel = np.diff(states_pca, axis=0)
        phys_vel = np.diff(positions[:, :3], axis=0)
        phys_vel_2d = phys_vel[:, :2]

        sv_norm = np.linalg.norm(state_vel, axis=1, keepdims=True) + 1e-8
        pv_norm = np.linalg.norm(phys_vel_2d, axis=1, keepdims=True) + 1e-8

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        ax1.quiver(states_pca[:-1, 0], states_pca[:-1, 1],
                   state_vel[:, 0] / sv_norm[:, 0], state_vel[:, 1] / sv_norm[:, 0],
                   color='#e74c3c', alpha=0.7, scale=15, width=0.005, headwidth=4)
        ax1.scatter(states_pca[:, 0], states_pca[:, 1],
                    c=np.arange(T), cmap='plasma', s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax1.plot(states_pca[:, 0], states_pca[:, 1], 'k-', alpha=0.2, linewidth=1)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('(a) Implicit Motion Trend in Mamba State Space\n'
                      'Arrows = State Transition Direction')
        ax1.grid(True, linestyle='--', alpha=0.4)

        ax2.quiver(positions[:-1, 0], positions[:-1, 1],
                   phys_vel_2d[:, 0] / pv_norm[:, 0], phys_vel_2d[:, 1] / pv_norm[:, 0],
                   color='#3498db', alpha=0.7, scale=15, width=0.005, headwidth=4)
        ax2.scatter(positions[:, 0], positions[:, 1],
                    c=np.arange(T), cmap='plasma', s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax2.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.2, linewidth=1)
        ax2.set_xlabel('Physical X (m)')
        ax2.set_ylabel('Physical Y (m)')
        ax2.set_title('(b) Explicit Motion Trend in Physical Space\n'
                      'Arrows = Physical Displacement Direction')
        ax2.grid(True, linestyle='--', alpha=0.4)

        plt.suptitle(f'Motion Trend Isomorphism: State Space vs Physical Space\n'
                     f'{env_name} sample {idx}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(sample_dir / 'qualitative_motion_trend_quiver.png', dpi=200, bbox_inches='tight')
        plt.close()

    # ---------- 3. 长期记忆衰减曲线 ----------
    if T >= 10:
        n_comp_5 = min(5, states.shape[0], states.shape[1])
        pca = PCA(n_components=n_comp_5)
        states_pca = pca.fit_transform(states)

        max_lag = min(T // 2, 20)
        autocorrs = []

        for lag in range(1, max_lag + 1):
            corrs = []
            for dim in range(states_pca.shape[1]):
                if np.std(states_pca[:-lag, dim]) > 1e-6 and np.std(states_pca[lag:, dim]) > 1e-6:
                    c = np.corrcoef(states_pca[:-lag, dim], states_pca[lag:, dim])[0, 1]
                    corrs.append(c)
            autocorrs.append(np.mean(corrs) if corrs else 0.0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_lag + 1), autocorrs, 'o-', color='#e74c3c',
                 linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
        plt.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong memory (r=0.5)')
        plt.xlabel('Time Lag (decision steps)')
        plt.ylabel('Mean Auto-correlation of Mamba States')
        plt.title(f'Long-Term Memory Decay in Mamba Hidden States\n'
                  f'{env_name} sample {idx} (higher = longer memory)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(sample_dir / 'qualitative_memory_decay.png', dpi=200, bbox_inches='tight')
        plt.close()

    # ---------- 4. 运动子空间投影 ----------
    n_comp_10 = min(10, states.shape[0], states.shape[1])
    pca = PCA(n_components=n_comp_10)
    states_pca = pca.fit_transform(states)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    main_actions = [1, 2, 3, 4]
    for ax_idx, act_id in enumerate(main_actions):
        if act_id not in actions:
            axes[ax_idx].set_visible(False)
            continue

        ax = axes[ax_idx]
        act_mask = actions == act_id
        act_indices = np.where(act_mask)[0]

        pre_states = []
        post_states = []
        for ai in act_indices:
            if ai > 0 and ai < T - 1:
                pre_states.append(states_pca[ai - 1, :2])
                post_states.append(states_pca[ai + 1, :2])

        if len(pre_states) == 0:
            axes[ax_idx].set_visible(False)
            continue

        pre_states = np.array(pre_states)
        post_states = np.array(post_states)

        ax.scatter(states_pca[:, 0], states_pca[:, 1], c='lightgray', s=30, alpha=0.3, zorder=1)

        for i in range(len(pre_states)):
            ax.annotate('', xy=post_states[i], xytext=pre_states[i],
                        arrowprops=dict(arrowstyle='->', color=action_colors.get(act_id, '#333'),
                                        lw=2.5, alpha=0.7))
            ax.scatter(pre_states[i, 0], pre_states[i, 1], c='white', s=80,
                       edgecolors=action_colors.get(act_id), linewidth=2, zorder=5, marker='o')
            ax.scatter(post_states[i, 0], post_states[i, 1], c=action_colors.get(act_id),
                       s=80, edgecolors='k', linewidth=1, zorder=5, marker='s')

        mean_pre = pre_states.mean(axis=0)
        mean_post = post_states.mean(axis=0)
        ax.annotate('', xy=mean_post, xytext=mean_pre,
                    arrowprops=dict(arrowstyle='->', color='red', lw=4, alpha=0.9))

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'Action "{action_names.get(act_id)}" State Subspace\n'
                     f'{len(pre_states)} occurrences, avg shift: {np.linalg.norm(mean_post - mean_pre):.3f}')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(['All states', 'Pre-action', 'Post-action'], loc='best', fontsize=8)

    plt.suptitle(f'Implicit Motion Subspaces: State Responses to Specific Actions\n'
                 f'{env_name} sample {idx}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(sample_dir / 'qualitative_motion_subspaces.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---------- 5. 状态-物理轨迹同步对比 ----------
    if T >= 5:
        n_comp_3 = min(3, states.shape[0], states.shape[1])
        pca = PCA(n_components=n_comp_3)
        states_3d = pca.fit_transform(states)

        fig = plt.figure(figsize=(18, 8))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(states_3d[:, 0], states_3d[:, 1], states_3d[:, 2],
                 'k-', alpha=0.3, linewidth=1)
        sc1 = ax1.scatter(states_3d[:, 0], states_3d[:, 1], states_3d[:, 2],
                          c=np.arange(T), cmap='plasma', s=60,
                          edgecolors='k', linewidth=0.5, depthshade=True)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax1.set_title('(a) Mamba Internal State Trajectory\n(PCA 3D projection)')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                 'k-', alpha=0.3, linewidth=1)
        sc2 = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                          c=np.arange(T), cmap='plasma', s=60,
                          edgecolors='k', linewidth=0.5, depthshade=True)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('(b) Physical Flight Trajectory\n(Actual drone movement)')

        cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.5, pad=0.1)
        cbar1.set_label('Decision Step')
        cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.1)
        cbar2.set_label('Decision Step')

        plt.suptitle(f'Trajectory Morphology Isomorphism\n'
                     f'{env_name} sample {idx}', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(sample_dir / 'qualitative_trajectory_isomorphism.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"[Qualitative] Saved attention/state visualizations for sample {idx}")

# ==================== 定性分析：注意力与状态可视化 ====================


# ==================== 全局运动语义可视化 ====================

def analyze_global_motion_manifold(all_samples_data, output_dir, env_name):
    """
    全局分析：将所有样本的所有时间步的 Mamba 状态一起降维，
    按物理运动类型（而非动作ID）着色，展示状态流形上的运动语义分离。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有状态并计算物理运动语义
    all_states = []
    all_motion_labels = []
    all_motion_names = []
    all_sample_ids = []
    all_time_steps = []

    for sample in all_samples_data:
        states = sample['states']
        actions = sample['actions']
        positions = sample['positions']
        sample_id = sample['sample_id']
        T = len(states)

        motion_labels, motion_names = compute_physical_motion_semantics(positions, actions)

        all_states.append(states)
        all_motion_labels.extend(motion_labels)
        all_motion_names.extend(motion_names)
        all_sample_ids.extend([sample_id] * T)
        all_time_steps.extend(range(T))

    all_states = np.vstack(all_states)
    all_motion_labels = np.array(all_motion_labels)
    all_sample_ids = np.array(all_sample_ids)
    all_time_steps = np.array(all_time_steps)

    N = len(all_states)
    print(f"Global manifold: {N} state vectors from {len(all_samples_data)} samples")

    # 全局 PCA 降维
    n_comp = min(2, all_states.shape[0], all_states.shape[1])
    if n_comp < 2:
        print("Not enough dimensions for 2D projection")
        return

    pca = PCA(n_components=n_comp)
    states_2d = pca.fit_transform(all_states)

    # 修复：正确拼写 explained_variance_ratio_
    evr = pca.explained_variance_ratio_

    motion_type_colors = {
        0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12',
        4: '#9b59b6', 5: '#1abc9c', 6: '#e67e22', 7: '#34495e',
        8: '#16a085', 9: '#d35400',
    }

    motion_type_names = {
        0: 'Hover/Stop', 1: 'Forward', 2: 'Turn Left', 3: 'Turn Right',
        4: 'Ascend', 5: 'Descend', 6: 'Shift Left', 7: 'Shift Right',
        8: 'Forward 6', 9: 'Forward 9',
    }

    # 图1: 全局运动语义流形
    plt.figure(figsize=(16, 12))
    for motion_type in sorted(np.unique(all_motion_labels)):
        mask = all_motion_labels == motion_type
        color = motion_type_colors.get(motion_type, '#95a5a6')
        name = motion_type_names.get(motion_type, f'Motion {motion_type}')

        plt.scatter(states_2d[mask, 0], states_2d[mask, 1],
                    c=color, label=f'{name} (n={np.sum(mask)})',
                    s=30, alpha=0.6, edgecolors='none', zorder=2)

    # 修复：正确拼写 explained_variance_ratio_
    plt.xlabel(f'Global PC1 ({evr[0]:.1%})', fontsize=13)
    plt.ylabel(f'Global PC2 ({evr[1]:.1%})', fontsize=13)
    plt.title(f'Global Motion Semantic Manifold\n'
              f'{env_name} | {len(all_samples_data)} samples | {N} timesteps\n'
              f'Colors = Physical Motion Type', fontsize=14)
    plt.legend(loc='best', fontsize=10, framealpha=0.9, markerscale=2)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(output_dir / 'global_motion_semantic_manifold.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 图2: 带标注版本
    plt.figure(figsize=(18, 14))
    plt.scatter(states_2d[:, 0], states_2d[:, 1],
                c='lightgray', s=10, alpha=0.3, edgecolors='none', zorder=1)

    np.random.seed(42)
    for motion_type in sorted(np.unique(all_motion_labels)):
        mask = all_motion_labels == motion_type
        color = motion_type_colors.get(motion_type, '#95a5a6')
        name = motion_type_names.get(motion_type, f'Motion {motion_type}')

        plt.scatter(states_2d[mask, 0], states_2d[mask, 1],
                    c=color, s=40, alpha=0.7, edgecolors='none', zorder=2)

        type_points = states_2d[mask]
        if len(type_points) > 0:
            center = type_points.mean(axis=0)
            dists = np.linalg.norm(type_points - center, axis=1)
            rep_idx = np.argmin(dists)
            rep_point = type_points[rep_idx]

            plt.annotate(name, xy=rep_point, xytext=(5, 5),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                         color='white', fontweight='bold', zorder=5)

    # 修复：正确拼写 explained_variance_ratio_
    plt.xlabel(f'Global PC1 ({evr[0]:.1%})', fontsize=13)
    plt.ylabel(f'Global PC2 ({evr[1]:.1%})', fontsize=13)
    plt.title(f'Motion Semantic Manifold with Labels\n'
              f'{env_name} | Physical motion patterns in Mamba states', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(output_dir / 'global_motion_manifold_annotated.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 图3: 按样本分色
    n_samples = len(all_samples_data)
    sample_cmap = plt.cm.get_cmap('tab20', n_samples)

    plt.figure(figsize=(16, 12))
    for i, sample_id in enumerate(sorted(np.unique(all_sample_ids))):
        mask = all_sample_ids == sample_id
        plt.scatter(states_2d[mask, 0], states_2d[mask, 1],
                    c=[sample_cmap(i)], label=f'Sample {sample_id}',
                    s=30, alpha=0.7, edgecolors='none')

    # 修复：正确拼写 explained_variance_ratio_
    plt.xlabel(f'Global PC1 ({evr[0]:.1%})', fontsize=13)
    plt.ylabel(f'Global PC2 ({evr[1]:.1%})', fontsize=13)
    plt.title(f'Cross-Sample State Consistency\nColors = Different Trajectories', fontsize=14)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(output_dir / 'global_manifold_by_sample.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 保存数据
    np.savez(output_dir / 'global_manifold_data.npz',
             states_2d=states_2d,
             motion_labels=all_motion_labels,
             motion_names=np.array(all_motion_names),
             sample_ids=all_sample_ids,
             time_steps=all_time_steps,
             pca_components=pca.components_,
             pca_explained_variance_ratio=evr)  # 修复：使用变量 evr

    print(f"[Global Manifold] Saved to {output_dir}")
    print(f"  Total states: {N}")
    print(f"  Motion distribution: {dict(zip(*np.unique(all_motion_labels, return_counts=True)))}")

def compute_physical_motion_semantics(positions, actions):
    """
    基于物理位置变化计算运动语义标签，而非直接使用动作ID。

    返回:
        motion_labels: List[int]  物理运动类型
        motion_names: List[str]   运动语义名称
    """
    T = len(positions)
    motion_labels = []
    motion_names = []

    for t in range(T):
        if t == 0:
            # 第一个时间步：无法计算运动，用动作ID推断
            motion_labels.append(int(actions[t]))
            motion_names.append(f"Action_{actions[t]}")
            continue

        # 计算物理位移
        dx = positions[t, 0] - positions[t - 1, 0]
        dy = positions[t, 1] - positions[t - 1, 1]
        dz = positions[t, 2] - positions[t - 1, 2]
        dyaw = positions[t, 3] - positions[t - 1, 3]

        # 归一化位移
        horiz_dist = np.sqrt(dx ** 2 + dy ** 2)
        total_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 判断运动语义
        if total_dist < 0.5 and abs(dyaw) < 0.1:
            motion_labels.append(0)
            motion_names.append("Hover/Stop")
        elif abs(dz) > horiz_dist * 0.5:
            if dz > 0:
                motion_labels.append(4)
                motion_names.append("Ascend")
            else:
                motion_labels.append(5)
                motion_names.append("Descend")
        elif abs(dyaw) > 0.2:  # ~11度
            if dyaw > 0:
                motion_labels.append(2)
                motion_names.append("Turn Left")
            else:
                motion_labels.append(3)
                motion_names.append("Turn Right")
        elif horiz_dist > 4.0:
            motion_labels.append(8)
            motion_names.append("Fast Forward")
        elif horiz_dist > 1.5:
            # 判断是前进还是平移
            yaw = positions[t - 1, 3]
            # 前进方向向量
            fwd_x = np.cos(yaw)
            fwd_y = np.sin(yaw)
            # 位移在前进方向的投影
            proj_fwd = dx * fwd_x + dy * fwd_y
            # 垂直于前进方向的分量
            proj_side = abs(dx * (-fwd_y) + dy * fwd_x)

            if proj_side > proj_fwd * 0.5:
                if dx * (-np.sin(yaw)) + dy * np.cos(yaw) > 0:
                    motion_labels.append(6)
                    motion_names.append("Shift Left")
                else:
                    motion_labels.append(7)
                    motion_names.append("Shift Right")
            else:
                motion_labels.append(1)
                motion_names.append("Forward")
        else:
            motion_labels.append(0)
            motion_names.append("Hover/Stop")

    return motion_labels, motion_names



# ==================== 新增函数结束 ====================


def main():
    # eval_info = "../configs/eval_test.json"
    # eval_info = "../configs/eval_test2.json"
    # eval_info = "../configs/eval_test_airsim_4.json"
    # eval_info = "../configs/eval.json"
    # eval_info = "../configs/airsim16_entries.json" # 3.96
    eval_info = "../configs/eval_test4.json" # 13.33

    f = open(eval_info, 'r')
    all_eval_info = json.loads(f.read())
    f.close()

    # 收集所有样本数据用于全局分析
    all_samples_data = []

    # Load model
    # model_name_or_path="/mnt/sdc/weiguanzhao/openfly-agent-7b"
    # processor = AutoProcessor.from_pretrained(model_name_or_path)
    # policy = AutoModelForVision2Seq.from_pretrained(
    #     model_name_or_path,
    #     attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # ).to("cuda:2")

    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila-llama3-8b-8f+b2+lr-0.0005+lora-r32+dropout-0.0"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b2+lr-0.0005+lora-r32+dropout-0.0"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln6/navila+b2+lr-0.0005+lora-r32+dropout-0.0"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"

    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"

    # Lora baseline ??? Baseline
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln-buffer-100000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln-buffer-100000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/vln-buffer-100000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+15000" # Now

    # This one is my result Mamba
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-buffer-10000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000" # Now
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-buffer-10000/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000" # Now
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000" # Now

    # Key
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+5000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key3/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7500" # Now
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key3/navila+b1+lr-0.0001+lora-r32+dropout-0.05+2500"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+15000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key2/navila+b1+lr-1e-05+lora-r32+dropout-0.0+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key/navila+b1+lr-0.0005+lora-r32+dropout-0.0+20000"

    # Key + Mamba
    model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-key-mamba/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7500"

    # KS
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+2000"

    # Random KS
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+2500"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+5000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7500"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-rand-ks/navila+b1+lr-0.0001+lora-r32+dropout-0.05+10000"

    # VTM
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-vtm/navila+b1+lr-0.0001+lora-r32+dropout-0.05+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-vtm/navila+b1+lr-0.0001+lora-r32+dropout-0.05+12500"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-vtm/navila+b1+lr-0.0001+lora-r32+dropout-0.05+15000"

    # Transformer
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-transformer/navila+b1+lr-0.0001+lora-r32+dropout-0.05+20000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-transformer/navila+b1+lr-0.0001+lora-r32+dropout-0.05+10000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-transformer/navila+b1+lr-0.0001+lora-r32+dropout-0.05+5000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-transformer/navila+b1+lr-0.0001+lora-r32+dropout-0.05+4000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-transformer/navila+b1+lr-0.0001+lora-r32+dropout-0.05+6000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-transformer/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7000"

    # K = 2
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-memory-2/navila+b1+lr-0.0001+lora-r32+dropout-0.05+3000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-memory-2/navila+b1+lr-0.0001+lora-r32+dropout-0.05+6000"

    # K = 8
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-memory-8/navila+b1+lr-0.0001+lora-r32+dropout-0.05+7000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-memory-8/navila+b1+lr-0.0001+lora-r32+dropout-0.05+6000"

    # model_path = "/mnt/sdc/weiguanzhao/navila-finetune/runs/navila+b1+lr-0.0005+lora-r32+dropout-0.0+10000"

    # Full text
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-fulltext/navila+b1+lr-0.0001+lora-r32+dropout-0.05+9000"
    # model_path = "/mnt/sdd/weiguanzhao/navila-finetune/runs-fulltext/navila+b1+lr-0.0001+lora-r32+dropout-0.05+3000"

    config = LlavaLlamaConfig.from_pretrained(model_path, resume=False)
    if getattr(config, "resume_path", None) is not None:
        config.resume_path = model_path

    model = LlavaLlamaModel(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=4096,
        # model_max_length=2048,
    ).to("cuda:3")

    # model = LlavaLlamaModel2(
    #     config=config,
    #     attn_implementation="flash_attention_2",
    #     model_max_length=4096,
    #     # model_max_length=2048,
    # ).to("cuda:1")

    model.eval()

    # processor = MultiModalProcessor(model)
    # norm_stats = None
    action_tokenizer = ActionTokenizer(model.tokenizer)

    # 自己训练的 pt 模型导入
    dataset_statistics_path = "/mnt/sdc/weiguanzhao/dataset_statistics.json"
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        # policy.norm_stats = norm_stats
        model.norm_stats = norm_stats

    # Test metrics
    acc = 0
    stop = 0
    data_num = 0
    MAX_STEP = 100

    # For overall metrics
    all_success = []
    all_osr = []
    all_ne = []
    all_spl = []

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:

        # 创建样本专用目录
        # sample_dir = Path(f"eval_vis/{env_name}/sample_{idx:03d}")
        # sample_dir.mkdir(parents=True, exist_ok=True)

        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)

    # Process each environment type sequentially
    for env_name, eval_info in env_groups.items():
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_info)} data entries")
        time.sleep(5)

        # Create appropriate environment bridge based on environment type
        if "airsim" in env_name:
            env_bridge = AirsimBridge(env_name)
            pos_ratio = 1.0
        elif "ue" in env_name:
            env_bridge = UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name)
            pos_ratio = 1.0
        elif "gs" in env_name:
            env_bridge = GSBridge(env_name)
            pos_ratio = 5.15
        else:
            print(f"Unknown environment type: {env_name}, skipping")
            continue

        # Evaluate all data for current environment
        for idx, item in enumerate(eval_info):
            acts = []  # Reset action list

            # 创建样本专用目录
            sample_dir = Path(f"eval_vis/{env_name}/sample_{idx:03d}")
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 存储轨迹数据（修复：正确收集每一步的 Mamba 状态）
            trajectory_states = []  # 每个元素: 当前决策步的 Mamba 状态向量 (D,)
            trajectory_actions = []  # 对应的动作 ID
            trajectory_positions = []  # 对应的物理位置 [x, y, z, yaw]

            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")

            stop_error = 1
            image_error = False

            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_postion[0] / pos_ratio,
                start_postion[1] / pos_ratio,
                start_postion[2] / pos_ratio,
                pitch,
                np.rad2deg(start_yaw),
                0
            )

            step = 0
            flag_osr = 0
            image_list = []
            env_bridge.pass_len = 10
            old_pose = new_pose
            data_num += 1
            same_seq = False

            while step < MAX_STEP:
                try:
                    raw_image = env_bridge.get_camera_data()
                    cv2.imwrite("test/cur_img.jpg", raw_image)
                    image = raw_image

                    image_list.append(image)
                    start = time.time()
                    same_seq = True

                    # ==================== 关键修改：正确收集 Mamba 状态 ====================
                    model_action, last50_seq = get_action3(
                        model, image_list, text, norm_stats, action_tokenizer,
                        same_seq, return_seq=True
                    )

                    # 提取当前决策步的 Mamba 状态：取序列最后一个时间步（最新状态）
                    if last50_seq is not None and last50_seq.ndim == 2:
                        current_state = last50_seq[-1, :]  # (D,)  ← 这是关键！
                        trajectory_states.append(current_state)
                        trajectory_actions.append(model_action)
                        trajectory_positions.append(new_pose.copy())
                    else:
                        print(f"Warning: No valid Mamba state at step {step}")
                    # ==================== 修改结束 ====================

                    end = time.time()
                    elapsed = end - start
                    acts.append(model_action)
                    same_seq = True
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, "
                          f"Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0] / pos_ratio,
                        new_pose[1] / pos_ratio,
                        new_pose[2] / pos_ratio,
                        pitch,
                        np.rad2deg(new_pose[3]),
                        0
                    )
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose

                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break

            if len(trajectory_states) >= 3:
                all_samples_data.append({
                    'states': np.vstack(trajectory_states),
                    'actions': np.array(trajectory_actions),
                    'positions': np.array(trajectory_positions),
                    'sample_id': idx,
                })
            # ==================== 关键修改：调用运动状态证明分析 ====================
            # if len(trajectory_states) >= 5:
                # prove_motion_capture(
                #     trajectory_states,
                #     trajectory_actions,
                #     trajectory_positions,
                #     sample_dir,
                #     env_name,
                #     idx
                # )
                # 定性可视化（新增）
                # visualize_mamba_attention_and_states(
                #     trajectory_states,
                #     trajectory_actions,
                #     trajectory_positions,
                #     sample_dir,
                #     env_name,
                #     idx
                # )
                # 新增 2D 时间投影
                # plot_2d_time_projection(
                #     trajectory_states, trajectory_actions, trajectory_positions,
                #     sample_dir, env_name, idx
                # )
            else:
                print(f"Sample {idx}: only {len(trajectory_states)} states collected, "
                      f"skip motion capture proof (need >= 5).")
            # ==================== 修改结束 ====================

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_postion)
            env_bridge.distance_to_goal.append(dis + 50)
            if dis < 20:
                acc += 1
                env_bridge.success.append(1)
                env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
            if flag_osr == 0:
                env_bridge.osr.append(0)
            env_bridge.print_info()

            all_success.append(env_bridge.success[-1])
            all_osr.append(env_bridge.osr[-1])
            all_ne.append(env_bridge.distance_to_goal[-1])
            all_spl.append(env_bridge.spl[-1])

            if image_error:
                continue

        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        del env_bridge
        import gc
        gc.collect()

    # 循环结束后调用全局分析
    if len(all_samples_data) > 0:
        analyze_global_motion_manifold(
            all_samples_data,
            Path(f"eval_vis/{env_name}/global_analysis"),
            env_name
        )

    # Final results
    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = 1 - stop / data_num if data_num > 0 else 0

    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final stop rate: {final_stop:.4f}")

    avg_sr = np.mean(all_success) if all_success else 0.0
    avg_osr = np.mean(all_osr) if all_osr else 0.0
    avg_ne = np.mean(all_ne) if all_ne else 0.0
    avg_spl = np.mean(all_spl) if all_spl else 0.0

    print(f"---")
    print(f"Average Success Rate (SR): {avg_sr:.4f}")
    print(f"Average Oracle Success Rate (OSR): {avg_osr:.4f}")
    print(f"Average Navigation Error (NE): {avg_ne:.4f}")
    print(f"Average Split Error (SPL): {avg_spl:.4f}")




if __name__ == '__main__':
    main()