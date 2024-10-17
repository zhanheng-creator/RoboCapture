# ### Demo for Robotics Transformer 1, with Training, Validation and Visualization Pipeline

# Cell
# uncomment if you are running on colab to mount your google drive
# from google.colab import drive
# drive.mount('/content/drive')

# When we are running codes in the colab, we need to mount google drives and read codes. Put the path to this repo's code at your google drive into this path

# Cell
import sys

# sys.path.append("path/to/this/code/repository/in/google/drive")

# Installing necessary dependencies

# Cell

# Import necessary dependencies

# Cell
import copy
import time
import random
from collections import OrderedDict
import func_timeout
from PIL import Image
from func_timeout import func_set_timeout
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gym import spaces
from skvideo.io import vwrite
import pybullet as p
import pybullet_data as pdata

from tqdm import tqdm, trange

import util.misc as utils
from IO_dataset_torch import build_dataset
from maruya24_rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from maruya24_rt1.transformer_network import TransformerNetwork
from maruya24_rt1.transformer_network_test_set_up import state_space_list

# Args for training, set your log directory in "log_dir", and adjust "batch_size" according to your gpu memory, and put downloaded dataset path in "data_path"
#如果改了动作空间，需要在demo测试文件和训练代码上都更改
#如果抖动，没夹到之后晃动，可能是动作空间的问题

# Cell
args = {
    "mode": "train",
    "device": "cuda:1",
    #测试改ceshi
    "data_path": "/data/yuantingyu/cube/",
    #测试改ceshi
    "cam_view": ["front", "wrist","side"],
    "log_dir": "logs",
    "time_sequence_length": 6,
    "lr": 0.0001,
    "batch_size": 6,
    "epochs": 50,
    "resume": False,
    "resume_from_checkpoint": "",
    "predicting_next_ts": True,
    "world_size": 1,
    "dist_url": "env://",
    "val_interval": 1,
    "num_val_threads": 25,
    "num_train_episode": 200,
    "num_val_episode": 10,
    "using_proprioception": False,
    "network_configs": {
        "vocab_size": 256,
        "token_embedding_size_per_image": 512,
        "language_embedding_size": 512,
        "num_layers": 8,
        "layer_size": 128,
        "num_heads": 8,
        "feed_forward_size": 512,
        "dropout_rate": 0.1,
        "crop_size": 236,
        "use_token_learner": True,
    },
    "scheduler_configs": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6, "verbose": True},
}

# Building the training and validation dataset

# Cell
train_dataset, val_dataset = build_dataset(
    data_path=args["data_path"],
    time_sequence_length=args["time_sequence_length"],
    predicting_next_ts=args["predicting_next_ts"],
    num_train_episode=args["num_train_episode"],
    num_val_episode=args["num_val_episode"],
    cam_view=args["cam_view"],
    language_embedding_size=args["network_configs"]["language_embedding_size"],
)

# Action space for the network, tokenizer need this to tokenize continuous actions into discrete tokens

# Cell
_action_space = spaces.Dict(
    OrderedDict(
        [
            ("terminate_episode", spaces.Discrete(4)),
            (
                "world_vector",
                spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32),
            ),
            (
                "rotation_delta",
                spaces.Box(
                    low=-np.pi / 10, high=np.pi / 10, shape=(3,), dtype=np.float32
                ),
            ),
            (
                "gripper_closedness_action",
                spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            ),
        ]
    )
)

# Initialize the network, noting that the network has number of image encoders equal to that of camera views, and token embedding size increase along with number of camera views

# Cell
# Initialize the TransformerNetwork based on specified configurations
network_configs = args["network_configs"]
# Modify network configuration based on specific settings
network_configs["time_sequence_length"] = args["time_sequence_length"]
network_configs["num_encoders"] = len(args["cam_view"])
network_configs["token_embedding_size"] = network_configs[
    "token_embedding_size_per_image"
] * len(args["cam_view"])
del network_configs["token_embedding_size_per_image"]
network_configs["using_proprioception"] = args["using_proprioception"]
network_configs["input_tensor_space"] = state_space_list()[0]
network_configs["output_tensor_space"] = _action_space
network = TransformerNetwork(**network_configs)
device = torch.device(args["device"])
network.to(device)

# Load pretrained network parameters here by setting "resume" to True and giving path of saved network to args["resume_from_checkpoint"]

# Cell
# Load model weights, optimizer, scheduler settings, resume from checkpoints if specified
# args["resume_from_checkpoint"] = "/root/autodl-tmp/log/1725561428/19-checkpoint.pth"
#测试改ceshi
args["resume_from_checkpoint"] = "/data/yuantingyu/cube_test/9-6-(all)-data-400/19-checkpoint.pth"
args["resume"] = True
if args["resume"]:
    checkpoint = torch.load(args["resume_from_checkpoint"], map_location="cpu")
    network.load_state_dict(checkpoint["model_state_dict"])
    print("loaded from: ", args["resume_from_checkpoint"])

total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print("number of model params:", total_params)
total_size_bytes = total_params * 4
# Parameter is in torch.float32，Each parameter takes 4 bytes
total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
print("model size: ", total_size_mb, " MB")

# Functions for sending all the tensor in dictionary to intended device, and retrieving the tensors at specific timestep of from a dictionary

# Cell
def dict_to_device(dict_obj, device):
    """
    put all the values in the [dict_obj] to [device]
    """
    for k, v in dict_obj.items():
        assert isinstance(v, torch.Tensor)
        dict_obj[k] = v.to(device)
    return dict_obj


def retrieve_single_timestep(dict_obj, idx):
    """
    get all the values in the [dict_obj] at index [idx]
    v[:, idx], all the values in the dictionary at second dimension needs to be same
    """
    dict_obj_return = copy.deepcopy(dict_obj)
    for k, v in dict_obj.items():
        dict_obj_return[k] = v[:, idx]
    return dict_obj_return

# Set training = True to start training, otherwise we can skip this block and using pretrained network to validate and visualize results

# Cell
training = False

if training:
    optimizer = torch.optim.AdamW(network.parameters(), lr=args["lr"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    for e in range(args["epochs"]):
        with tqdm(train_dataloader, dynamic_ncols=True, desc="train") as tqdmDataLoader:
            for _, (obs, action) in enumerate(tqdmDataLoader):
                optimizer.zero_grad()
                network.set_actions(dict_to_device(action, device))
                network_state = np_to_tensor(
                    batched_space_sampler(
                        network._state_space,
                        batch_size=args["batch_size"],
                    )
                )
                output_actions, network_state = network(
                    dict_to_device(obs, device),
                    dict_to_device(network_state, device),
                )

                loss = network.get_actor_loss().mean()

                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": e,
                        "loss": loss.item(),
                        "gpu_memory_used": str(
                            round(torch.cuda.max_memory_allocated() / (1024**3), 2)
                        )
                        + " GB",
                        "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )

# Set validating = False if you don't want to validate results, this function will output a set of images, each contains 8 subplots(corresponding to 8 action tokens output of model), each drawn with ground truth and output from model, on the title, the number marks the total loss value of this action.

# Cell
validating = False


def visualize(all_gt, all_output):
    all_output = all_output[:, -1, :]
    all_gt = all_gt[:, -1, :]
    title = [
        "terminate_episode_l1_error: ",
        "cmd_pos_x_l1_error: ",
        "cmd_pos_y_l1_error: ",
        "cmd_pos_z_l1_error: ",
        "cmd_rot_x_l1_error: ",
        "cmd_rot_y_l1_error: ",
        "cmd_rot_z_l1_error: ",
        "cmd_gripper_l1_error: ",
    ]
    plt.figure(figsize=(22, 12))
    for i in range(8):
        c = utils.generate_random_color()
        plt.subplot(2, 4, i + 1)
        val_loss = F.l1_loss(
            torch.from_numpy(all_output[:, i]).float(),
            torch.from_numpy(all_gt[:, i]).float(),
        )
        plt.title(title[i] + str(val_loss.cpu().data.numpy()))
        plt.plot(all_gt[:, i], c=c, label="gt")
        plt.plot(all_output[:, i], c=c, linestyle="dashed", label="output")
        plt.xlabel("timesteps")
        plt.ylabel("action_tokens")
        plt.grid()
        plt.legend()
    plt.show()


val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
network.eval()
gt_one_episode = []
model_output_one_episode = []

if validating:
    for idx, (obs, action) in tqdm(
        enumerate(val_dataloader), desc="validation", total=len(val_dataset)
    ):
        # Initialize network state
        network_state = batched_space_sampler(network._state_space, batch_size=1)
        network_state = np_to_tensor(network_state)

        # Reset network state
        for k, v in network_state.items():
            network_state[k] = torch.zeros_like(v)

        action_predictions_logits = []
        output_actions = []

        for i_ts in range(args["time_sequence_length"]):
            ob = retrieve_single_timestep(obs, i_ts)
            output_action, network_state = network(
                dict_to_device(ob, device),
                dict_to_device(network_state, device),
            )
            output_actions.append(output_action)
            action_predictions_logits.append(
                network._aux_info["action_predictions_logits"]
            )

        # Get ground truth actions
        gt_actions = network._action_tokenizer.tokenize(action)
        action_predictions_logits = (
            torch.cat(action_predictions_logits, dim=0).unsqueeze(0).permute(0, 3, 1, 2)
        )
        gt_one_episode.append(gt_actions)
        model_output_one_episode.append(action_predictions_logits.argmax(1))

        # Handle end of episode scenario
        if gt_actions[0, -1, 0] == 2:
            # gt_actions[0, -1, 0] is the terminate signal for current episode, 2 indicates the end of episode
            # whtn terminate signal is triggered, we write this episode's test results into files
            gt_one_episode = torch.cat(gt_one_episode).cpu().data.numpy()
            model_output_one_episode = (
                torch.cat(model_output_one_episode).cpu().data.numpy()
            )

            # Visualize and store episode results

            visualize(gt_one_episode, model_output_one_episode)
            gt_one_episode = []
            model_output_one_episode = []

# The following 3 blocks contains utilities for visualization on model results

# Cell
import threading

TASKS = {"touch": "TouchTaskEnv", "pick": "PickTaskEnv"}

CAM_INFO = {
    "front": [[0, 0, 0.7], 1.8, 180, -20, 0, 40],
    "fronttop": [[0, 0.5, 0.7], 1.5, 180, -60, 0, 35],
    "topdown": [[0, 0.35, 0], 2.0, 0, -90, 0, 45],
    "side": [[0, 0.35, 0.9], 1.5, 90, 0, 0, 40],
    # "root": [[0, 1.3, 0.85], 1.5, 0, -10, 0, 90],
    "root": [[0, 0.6, 0.75], 1.3, -35, -5, 0, 40],
    "wrist": [],
}
cam_resolution = (1080, 864)


def get_cam_projection_matrix(cam_view):
    # print(fov)
    aspect = float(cam_resolution[0]) / cam_resolution[1]
    nearVal = 0.1
    farVal = 100

    if cam_view == "wrist":
        fov = 100
        nearVal = 0.018
    else:
        fov = CAM_INFO[cam_view][-1]
    cam_projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal,
    )
    return cam_projection_matrix


def get_view_matrix(cam_view, robot_id, ee_index):
    if cam_view == "wrist":
        eye_pos, eye_ori = p.getLinkState(
            robot_id,
            ee_index,
            computeForwardKinematics=True,
        )[0:2]
        eye_pos = list(eye_pos)
        eye_pos = p.multiplyTransforms(eye_pos, eye_ori, [0, 0, -0.05], [0, 0, 0, 1])[0]
        r_mat = p.getMatrixFromQuaternion(eye_ori)
        tx_vec = np.array([r_mat[0], r_mat[3], r_mat[6]])
        ty_vec = np.array([r_mat[1], r_mat[4], r_mat[7]])
        tz_vec = np.array([r_mat[2], r_mat[5], r_mat[8]])
        camera_position = np.array(eye_pos)
        target_position = eye_pos + 0.001 * tz_vec
        cam_view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=ty_vec,
        )
    else:
        cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            CAM_INFO[cam_view][0],
            CAM_INFO[cam_view][1],
            CAM_INFO[cam_view][2],
            CAM_INFO[cam_view][3],
            CAM_INFO[cam_view][4],
            2,
        )
    return cam_view_matrix


def get_cam_view_img(cam_view, robot_id=None, ee_index=None):
    cam_view_matrix = get_view_matrix(cam_view, robot_id, ee_index)
    cam_projection_matrix = get_cam_projection_matrix(cam_view)
    (width, height, rgb_pixels, _, _) = p.getCameraImage(
        cam_resolution[0],
        cam_resolution[1],
        viewMatrix=cam_view_matrix,
        projectionMatrix=cam_projection_matrix,
    )
    rgb_array = np.array(rgb_pixels).reshape((height, width, 4)).astype(np.uint8)
    img = np.array(resize_and_crop(rgb_array[:, :, :3]))
    return img


def resize_and_crop(input_image):
    """Crop the image to 5:4 aspect ratio and resize it to 320x256 pixels."""
    input_image = Image.fromarray(input_image)
    width, height = input_image.size
    target_aspect = 5 / 4
    current_aspect = width / height

    if current_aspect > target_aspect:
        # If the image is too wide, crop its width
        new_width = int(target_aspect * height)
        left_margin = (width - new_width) / 2
        input_image = input_image.crop((left_margin, 0, width - left_margin, height))
    elif current_aspect < target_aspect:
        # If the image is too tall, crop its height
        new_height = int(width / target_aspect)
        top_margin = (height - new_height) / 2
        input_image = input_image.crop((0, top_margin, width, height - top_margin))

    # Resize the image to 320x256
    input_image = input_image.resize((320, 256))
    return input_image


class Panda(object):
    def __init__(self):
        self.arm_dof = 7
        self.ee_index = 11
        self.home_j_pos = [1.22, -0.458, 0.31, -2.0, 0.20, 1.56, 2.32, 0.04, 0.04]

    def load(self):
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [0, 0, 0.62],
            [0, 0, 0, 1],
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )

        # create a constraint to keep the fingers centered, 9 and 10 for finger indices
        c = p.createConstraint(
            self.robot_id,
            9,
            self.robot_id,
            10,
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def calc_ik(self, pose):
        return list(
            p.calculateInverseKinematics(
                self.robot_id,
                self.ee_index,
                pose[0],
                pose[1],
                [-7] * 7,
                [7] * 7,
                [7] * 7,
                self.home_j_pos,
                maxNumIterations=100,
                solver=p.IK_DLS,
            )
        )

    def reset_j_home(self, random_home=False):
        if random_home:
            for i in range(self.arm_dof):
                self.home_j_pos[i] += random.uniform(-np.pi / 10, np.pi / 10)
        self.reset_j_pos(self.home_j_pos)

    def reset_j_pos(self, j_pos):
        index = 0
        for j in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)
            joint_type = p.getJointInfo(self.robot_id, j)[2]
            if joint_type in [
                p.JOINT_PRISMATIC,
                p.JOINT_REVOLUTE,
            ]:
                p.resetJointState(self.robot_id, j, j_pos[index])
                index = index + 1

    def move_j(self, j_pos):
        for i in range(self.arm_dof):
            max_vel = 1 if i == self.arm_dof - 1 else 0.5
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                j_pos[i],
                maxVelocity=max_vel,
                force=5 * 240.0,
            )
        for i in [9, 10]:
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                j_pos[-1],
                maxVelocity=0.02,
                force=1000,
            )

    def is_j_arrived(self, j_pos, include_finger=True, threshold=1e-2):
        cur_joint_position = [
            s[0]
            for s in p.getJointStates(
                self.robot_id, list(range(self.arm_dof)) + [9, 10]
            )
        ]
        diff_arm = np.abs(np.array(cur_joint_position) - np.array(j_pos))
        is_arrive = np.all(diff_arm[: self.arm_dof - 1] <= threshold)
        if include_finger:
            is_arrive = is_arrive and np.all(diff_arm[-2:] <= threshold)
        return is_arrive

    def is_ee_arrived(self, ee_pose, tar_obj_id=None, threshold=2 * 1e-2):
        panda_ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_index)[0])
        diff_pos = np.abs(panda_ee_pos - ee_pose[0])
        is_arrive = np.all(diff_pos <= threshold)
        if tar_obj_id != None:
            is_arrive = is_arrive and p.getClosestPoints(
                self.robot_id, tar_obj_id, 1e-5
            )
            # p.getContactPoints(self.robot_id, tar_obj_id)
        return is_arrive


class SimEnv(object):
    def __init__(self):
        p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.control_dt = 1.0 / 240.0
        self.reset_env_wait_time = 0.5
        self.robot = None
        self.tar_obj = None
        self.tar_obj_pose = None
        self.state = None
        self.target_waypoints = None
        self.data_record_fq = None
        self.collected_traj = 700
        self.load_env()
        self.set_camera()
        self.reset_env()
        self.lock = threading.Lock()

    def load_env(self):
        raise NotImplementedError

    def reset_env(self):
        raise NotImplementedError

    def set_camera(self):
        # set camera configs
        # self.cam_resolution = (320, 256)
        self.cam_resolution = (1080, 864)
        self.cam_info = CAM_INFO
        self.cam_view_matrice = []
        for key, val in self.cam_info.items():
            if key == "wrist":
                self.cam_view_matrice.append([])
            else:
                self.cam_view_matrice.append(
                    p.computeViewMatrixFromYawPitchRoll(
                        val[0],
                        val[1],
                        val[2],
                        val[3],
                        val[4],
                        2,
                    )
                )

        self.cam_projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.cam_resolution[0]) / self.cam_resolution[1],
            nearVal=0.1,
            farVal=100,
        )


class TouchTaskEnv(SimEnv):
    def __init__(self):
        super().__init__()

    def load_env(self):
        p.loadURDF("table/table.urdf", [0, 0.35, 0], [0, 0, 0, 1])
        #测试改ceshi
        self.tar_obj = p.loadURDF("urdf/cube/cube.urdf", [0, 0, 0], globalScaling=0.04)
        p.changeVisualShape(self.tar_obj, -1, rgbaColor=[1, 0, 0, 1])
        self.robot = Panda()
        self.robot.load()

    def reset_tar_obj(self, tar_obj_range=None, tar_pos_rot=None, random_pos_rot=True):
        if random_pos_rot:
            x = random.uniform(tar_obj_range[0], tar_obj_range[1])
            y = random.uniform(tar_obj_range[2], tar_obj_range[3])
            r = random.uniform(tar_obj_range[4], tar_obj_range[5])
        else:
            x = tar_pos_rot[0]
            y = tar_pos_rot[1]
            r = tar_pos_rot[2]
        #固定物体位置
        #测试改ceshi
        pos = [-0.28, 0.48, 0.645]
        rot = p.getQuaternionFromEuler([0, np.pi / 2, r])
        p.resetBasePositionAndOrientation(
            self.tar_obj,
            pos,
            rot,
        )
        self.tar_obj_pose = p.getBasePositionAndOrientation(self.tar_obj)

    def reset_env(self):
        self.robot.reset_j_home()
        time.sleep(1)
        self.state = 0
        self.t = 0
        self.state_stuck_t = 0


class PickTaskEnv(TouchTaskEnv):
    def __init__(self):
        super().__init__()

# Cell
def inference(network, imgs):
    network_state = batched_space_sampler(network._state_space, batch_size=1)
    network_state = np_to_tensor(
        network_state
    )  # change np.ndarray type of sample values into tensor type
    for k, v in network_state.items():
        network_state[k] = torch.zeros_like(v).to(device)
    output_actions = []
    obs = dict()
    obs["image"] = torch.stack(imgs, dim=1).to(device)
    obs["natural_language_embedding"] = torch.ones(1, 6, 512).to(device)
    with torch.no_grad():
        for i_ts in range(6):
            ob = retrieve_single_timestep(obs, i_ts)
            output_action, network_state = network(ob, network_state)
            output_actions.append(output_action)
        action = output_actions[-1]
    action = dict_to_device(action, torch.device("cpu"))
    return [
        action["terminate_episode"].flatten().tolist(),
        action["world_vector"].flatten().tolist(),
        action["rotation_delta"].flatten().tolist(),
        action["gripper_closedness_action"].flatten().tolist(),
    ]
    
from scipy.spatial.transform import Rotation as R
def add_quaternions(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    combined_rotation = r1 * r2  # 旋转合成
    return combined_rotation.as_quat()
# Cell
class SimTester:
    def __init__(self, task_name):
        p.connect(p.DIRECT)
        # p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraPitch=-20,
            cameraYaw=180,
            cameraTargetPosition=[0, 0, 0.6],
        )
        p.setAdditionalSearchPath(pdata.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.task_env = globals()[TASKS[task_name]]()
        self.max_step = 100
        self.device = torch.device("cuda")

    def test_step(self, delta_ee_pos, delta_ee_rot, gripper_cmd, cam_view):
        try:
            # start = time.time()
            self.execute_action(delta_ee_pos, delta_ee_rot, gripper_cmd, cam_view)
            # print(time.time() - start)
        except func_timeout.exceptions.FunctionTimedOut:
            # print("time out for execute actions")
            pass
        self.update_imgs(cam_view)

    @func_set_timeout(0.5)
    def execute_action(
        self, delta_ee_pos, delta_ee_rot, gripper_cmd, cam_view, relative=True
    ):
        if relative:
            if self.last_ee_pose == None:
                last_ee_pose = p.getLinkState(
                    self.task_env.robot.robot_id, self.task_env.robot.ee_index
                )[0:2]
            else:
                last_ee_pose = self.last_ee_pose
            # delta_ee_pos = [0, 0, 0.05]
            cur_ee_pose = p.multiplyTransforms(
                last_ee_pose[0],
                last_ee_pose[1],
                delta_ee_pos,
                p.getQuaternionFromEuler(delta_ee_rot),
            )
        else:
            raise NotImplementedError
        print("cur_ee_pose[0]")
        print(cur_ee_pose[0])
        print("cur_ee_pose[1]")
        print(cur_ee_pose[1])
        tar_j_pos = self.task_env.robot.calc_ik([cur_ee_pose[0], cur_ee_pose[1]])
        if gripper_cmd[0] > 0 or self.gripper_triggered:
            tar_j_pos[-2:] = [0.01] * 2  # close gripper
            if not self.gripper_triggered:
                self.gripper_triggered = True
        else:
            tar_j_pos[-2:] = [0.04] * 2  # open gripper

        self.task_env.robot.move_j(tar_j_pos)
        while not (self.task_env.robot.is_j_arrived(tar_j_pos, threshold=1e-3)):
            p.stepSimulation()
            time.sleep(0.005)




    def reset_tester(self, cam_view):
        self.task_env.reset_env()
        self.reset_imgs(cam_view)
        self.last_ee_pose = None
        self.gripper_triggered = False
        self.episode_succ = [False, False]

    def reset_imgs(self, cam_view):
        self.imgs = [torch.zeros(1, 3, 256 * len(cam_view), 320)] * 6

    def update_imgs(self, cam_view):
        img = self.get_obs_img(cam_view)
        self.imgs.pop(0)
        self.imgs.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0))

    def get_obs_img(self, cam_view):
        imgs = []
        for cview in cam_view:
            imgs.append(
                get_cam_view_img(
                    cview, self.task_env.robot.robot_id, self.task_env.robot.ee_index
                )
            )
        cur_img = np.concatenate(imgs, axis=0)
        return cur_img / 255.0

    def get_step_dist(self):
        c_point = p.getClosestPoints(
            self.task_env.robot.robot_id,
            self.task_env.tar_obj,
            distance=10,
            linkIndexA=10,
        )
        try:
            dist = min(np.array(c_point, dtype=object)[:, 8])
        except:
            return 0
        return dist

    def check_episode_succ(self):
        ee_z_pos = p.getLinkState(
            self.task_env.robot.robot_id, self.task_env.robot.ee_index
        )[0][2]
        tar_obj_z_pos = self.task_env.tar_obj_pose[0][2]
        contact_points = p.getContactPoints(
            self.task_env.robot.robot_id, self.task_env.tar_obj
        )
        if abs(ee_z_pos - tar_obj_z_pos) < 0.035 and len(contact_points) > 0:
            self.episode_succ[0] = True
        if ee_z_pos - tar_obj_z_pos > 0.1 and len(contact_points) > 0:
            self.episode_succ[1] = True
        return ee_z_pos - tar_obj_z_pos


import os
import pandas as pd
import time
import numpy as np
from tqdm import trange
from IPython.display import Video
# Visualizing model results, now the target object(red cube) is uniformly random-generated
# at the area with x(meter): [-0.3, 0.3], y(meter): [-0.3, 0.3] and rotation: [-np.pi / 2, np.pi / 2]
#
# if you want to assign a cube position, you can change random_pos_rot to False and use
# "sim_tester.task_env.reset_tar_obj(tar_pos_rot = tar_pos_rot, random_pos_rot = False)" where tar_pos_rot is a 3 element list contains [x, y, rotation]
#两边均匀采集
def get_random_tar_obj_range():
    ranges = [
        [0.15, 0.4, 0.4, 0.7, -np.pi / 2, np.pi / 2],
        [-0.4, -0.15, 0.4, 0.7, -np.pi / 2, np.pi / 2]
    ]
    return random.choice(ranges)
results = []
#测试改ceshi
video_dir = "/data/yuantingyu/cube_test/9-6-(all)-data-400/videos/"
os.makedirs(video_dir, exist_ok=True)

# Create a DataFrame to store results
results = []
#调整次数
for collect_num in range(0, 50):
    network.eval()
    if p.isConnected():
        p.disconnect()
    sim_tester = SimTester("pick")
    tar_obj_range = get_random_tar_obj_range()
    sim_tester.reset_tester(args["cam_view"])
    sim_tester.task_env.reset_tar_obj(tar_obj_range=tar_obj_range, random_pos_rot=True)
    sim_tester.update_imgs(args["cam_view"])
    vids = []
    
    for i in trange(55):
        imgs = sim_tester.imgs
        start = time.time()
        (
            terminate_episode,
            delta_ee_pos,
            delta_ee_rot,
            gripper_cmd,
        ) = inference(network, imgs)
        # delta_ee_rot = [0,0,0]

        sim_tester.test_step(delta_ee_pos, delta_ee_rot, gripper_cmd, args["cam_view"])
        vids.append(imgs[-1][0].permute(1, 2, 0).numpy() * 255)

    # contact_points = p.getContactPoints(bodyA=sim_tester.task_env.robot.robot_id, 
    #                                     bodyB=sim_tester.task_env.tar_obj)
    # success = bool(contact_points)
    # success_message = f"{collect_num}组机械臂成功抓取到了" if success else f"{collect_num}组机械臂未能抓取到"
    # print(success_message)
    #增加判断条件，物体需要离开桌面
    contact_points = p.getContactPoints(bodyA=sim_tester.task_env.robot.robot_id, 
                                    bodyB=sim_tester.task_env.tar_obj)
    success = bool(contact_points)

    # 获取目标物体底部的位置
    obj_pos, obj_orientation = p.getBasePositionAndOrientation(sim_tester.task_env.tar_obj)
    table_height = 0.645  # 将此处替换为物体高度
    some_threshold_value = 0.01
    object_bottom_height = obj_pos[2]  # 物体底部的 z 坐标

    # 检查物体底部是否离开了桌面
    object_lifted = object_bottom_height > table_height + some_threshold_value

    # 更新成功条件
    success = success and object_lifted

    success_message = f"{collect_num}组机械臂成功抓取到了" if success else f"{collect_num}组机械臂未能抓取到"
    print(success_message)

    #保存视频
    # Save the result to the DataFrame
    results.append({'Collect Num': collect_num, 'Success': success})
    # Save video
    video_path = os.path.join(video_dir, f"vis_{collect_num:03d}_组_19_checkpoint_9_6.mp4")
    vwrite(video_path, vids)
    Video(video_path, embed=True, width=320, height=512)

# Save results to Excel
df_results = pd.DataFrame(results)
#测试改ceshi
excel_path = "/data/yuantingyu/cube_test/9-6-(all)-data-400/(all)-data-400-19pth-results.xlsx"
df_results.to_excel(excel_path, index=False)

# Calculate success rate
success_count = df_results['Success'].sum()
total_attempts = len(df_results)
success_rate = success_count / total_attempts * 100

# Save success rate to the Excel file
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
    success_rate_df = pd.DataFrame({'Success Rate (%)': [success_rate]})
    success_rate_df.to_excel(writer, sheet_name='Success Rate', index=False)

# for collect_num in range(0,10):

#     # Cell
#     network.eval()
#     if p.isConnected():
#         p.disconnect()
#     sim_tester = SimTester("pick")
#     # sim_tester.reset_tester(args['cam_view'])
#     # tar_obj_range = [-0.3, 0.3, 0.4, 0.7, -np.pi / 2, np.pi / 2]
#     tar_obj_range = get_random_tar_obj_range()
#     sim_tester.reset_tester(args["cam_view"])
#     sim_tester.task_env.reset_tar_obj(tar_obj_range=tar_obj_range, random_pos_rot=True)
#     sim_tester.update_imgs(args["cam_view"])
#     vids = []
#     for i in trange(50):
#         imgs = sim_tester.imgs
#         start = time.time()
#         (
#             terminate_episode,
#             delta_ee_pos,
#             delta_ee_rot,
#             gripper_cmd,
#         ) = inference(network, imgs)
#         print("step")
#         print(i)
#         print("delta_ee_pos")
#         print(delta_ee_pos)
#         print("delta_ee_rot")
#         print(delta_ee_rot)
#         delta_ee_rot = [0,0,0]
#         print("gripper_cmd")
#         print(gripper_cmd)

#         sim_tester.test_step(delta_ee_pos, delta_ee_rot, gripper_cmd, args["cam_view"])
#         vids.append(imgs[-1][0].permute(1, 2, 0).numpy() * 255)

#     contact_points = p.getContactPoints(bodyA=sim_tester.task_env.robot.robot_id, 
#                                     bodyB=sim_tester.task_env.tar_obj)
#     if contact_points:
#         print(f"{collect_num}组机械臂成功抓取到了")
#     else:
#         print(f"{collect_num}组机械臂未能抓取到")


#     # visualize
#     from IPython.display import Video
#     vwrite("vis_7_3_checkpoint_9_4.mp4", vids)
#     Video("vis_7_3_checkpoint_9_4.mp4", embed=True, width=320, height=512)
