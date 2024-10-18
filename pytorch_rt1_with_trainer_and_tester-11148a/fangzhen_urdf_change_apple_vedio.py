# ### Demo for Robotics Transformer 1, with Training, Validation and Visualization Pipeline

# Cell
# uncomment if you are running on colab to mount your google drive
# from google.colab import drive
# drive.mount('/content/drive')

# When we are running codes in the colab, we need to mount google drives and read codes. Put the path to this repo's code at your google drive into this path

# Cell
import sys

sys.path.append("path/to/this/code/repository/in/google/drive")

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

# Cell
args = {
    "mode": "train",
    "device": "cuda:0",
    "data_path": "/home/taizun/111/pytorch_rt1_with_trainer_and_tester-11148a/Panda_pick",
    "cam_view": ["topdown"],
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
                spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32),
            ),
            (
                "rotation_delta",
                spaces.Box(
                    low=-np.pi / 5, high=np.pi / 5, shape=(3,), dtype=np.float32
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
args["resume_from_checkpoint"] = "/home/taizun/111/pytorch_rt1_with_trainer_and_tester-11148a/log/1724752649/14-checkpoint.pth"
args["resume"] = False
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
        print("fov")
        print(fov)
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
        # print("eye_pos")
        # print(eye_pos)
        # print("eye_ori")
        # print(eye_ori)
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
        # print("robot_id")
        # print(robot_id)
        # eye_pos, eye_ori = p.getLinkState(
        #     robot_id,
        #     ee_index,
        #     computeForwardKinematics=True,
        # )[0:2]
        # eye_pos = list(eye_pos)
        # eye_pos = p.multiplyTransforms(eye_pos, eye_ori, [0, 0, 0], [0, 0, 0, 1])[0]
        # print("eye_pos")
        # print(eye_pos)
        # print("eye_ori")
        # print(eye_ori)

    #---------------------
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

def get_cam_matrix(cam_view, robot_id=None, ee_index=None):
    cam_view_matrix = get_view_matrix(cam_view, robot_id, ee_index)
    cam_projection_matrix = get_cam_projection_matrix(cam_view)
    return cam_view_matrix, cam_projection_matrix

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
        aspect=float(self.cam_resolution[0]) / self.cam_resolution[1]
        print("aspect")
        print(aspect)


class TouchTaskEnv(SimEnv):
    def __init__(self):
        super().__init__()

    #加载环境（桌子、木块）
    #pink_tea_box  ok   rgbaColor=[1, 0.75, 0.8, 1]
    def load_env(self):
        p.loadURDF("table/table.urdf", [0, 0.35, 0], [0, 0, 0, 1])
        self.tar_obj = p.loadURDF("urdf/plastic_apple/model.urdf", [0, 0, 0], globalScaling=0.75)
        p.changeVisualShape(self.tar_obj, -1)
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
        #设置物体初始高度    一开始是0.645
        pos = [x, y, 0.645]
        #rot = p.getQuaternionFromEuler([0, np.pi / 2, r])
        rot = p.getQuaternionFromEuler([0, 0, r])
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

    #@func_set_timeout(0.2)
    @func_set_timeout(0.5)
    def execute_action(
        self, delta_ee_pos, delta_ee_rot, gripper_cmd, cam_view, relative=True
    ):
        if relative:
            if self.last_ee_pose == None:
                last_ee_pose = p.getLinkState(
                    self.task_env.robot.robot_id, self.task_env.robot.ee_index
                )[0:2]
                print("self.task_env.robot.robot_id")
                print(self.task_env.robot.robot_id)
                print("self.task_env.robot.ee_index")
                print(self.task_env.robot.ee_index)
                print("last_ee_pose")
                print(last_ee_pose)

            else:
                last_ee_pose = self.last_ee_pose

            # cur_ee_pose = p.multiplyTransforms(
            #     last_ee_pose[0],
            #     last_ee_pose[1],
            #     delta_ee_pos,
            #     p.getQuaternionFromEuler(delta_ee_rot),
            # )
            cur_ee_pose_pos = (
                last_ee_pose[0][0] + delta_ee_pos[0],
                last_ee_pose[0][1] + delta_ee_pos[1],
                last_ee_pose[0][2] + delta_ee_pos[2],
            )

            # 如果你不需要方向部分，可以忽略下面的部分
            # 如果需要保持方向不变，可以直接使用 last_ee_pose[1],last_ee_pose[1]是四元数
            cur_ee_pose_rot = list(add_quaternions(p.getQuaternionFromEuler(delta_ee_rot),last_ee_pose[1]) )# 保持方向不变

            # 合并位置和方向得到新的位姿
            cur_ee_pose = (cur_ee_pose_pos, cur_ee_pose_rot)
            print("执行函数里的last_ee_pose")
            print(last_ee_pose[0])
            print("执行函数里的delta_ee_pose")
            print(delta_ee_pos)
            print("cur_ee_pose")
            print(cur_ee_pose)
        else:
            raise NotImplementedError
        tar_j_pos = self.task_env.robot.calc_ik([cur_ee_pose[0], cur_ee_pose[1]])
        print("tar_j_pos")
        print(tar_j_pos)
        if gripper_cmd[0] > 0 or self.gripper_triggered:
            print("gripper_cmd[0] > 0 控制夹爪关闭")
            tar_j_pos[-2:] = [0.01] * 2  # close gripper
            if not self.gripper_triggered:
                self.gripper_triggered = True
        else:
            tar_j_pos[-2:] = [0.04] * 2  # open gripper

        self.task_env.robot.move_j(tar_j_pos)
        # if(gripper_cmd[0] > 0):
        #     threshold=2e-3
        # else:
        #    threshold=1e-3
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
            print("cam_view")
            print(cam_view)
            imgs.append(
                get_cam_view_img(
                    cview, self.task_env.robot.robot_id, self.task_env.robot.ee_index
                )
            )
        cur_img = np.concatenate(imgs, axis=0)
        return cur_img / 255.0
    
    def get_matrix(self, cam_view):
        viewMatrix , projectionMatrix = get_cam_matrix(cam_view,self.task_env.robot.robot_id, self.task_env.robot.ee_index)
        return viewMatrix , projectionMatrix

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


#将用于 pybullet 和 OpenGL 的4x4视图矩阵转换为在 ROS-TF 和 OpenCV 中常用的姿态（姿态由四元数表示）和位置（平移向量表示）
from scipy.spatial.transform import Rotation as R
import numpy as np
def bulletView2cvPose(viewMatrix):
    """
    bulletView2cvPose converts a 4x4 view matrix as used in 
    pybullet and openGL back to orientation and position as used 
    in ROS-TF and OpenCV.
    
    :param viewMatrix: 4x4 view matrix as used in pybullet and openGL
    :return: 
        q: ROS orientation expressed as quaternion [qx, qy, qz, qw]
        t: ROS position expressed as [tx, ty, tz]
    """
    
    # Reshape viewMatrix back to a 4x4 matrix
    T = np.array(viewMatrix).reshape(4, 4).T
    
    # Coordinate transformation matrix (reverse of what was applied)
    Tc = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]]).reshape(4,4)
    
    # Reverse the transformation applied in cvPose2BulletView
    T = np.linalg.inv(Tc @ T)
    
    # Extract rotation matrix R and translation vector t
    R_matrix = T[:3, :3]
    t = T[:3, 3]
    
    # Convert rotation matrix to quaternion
    q = R.from_matrix(R_matrix).as_quat()
    
    # Reorder quaternion back to ROS convention [qx, qy, qz, qw]
    q = [q[1], q[2], q[3], q[0]]
    
    return q, t



# Visualizing model results, now the target object(red cube) is uniformly random-generated
# at the area with x(meter): [-0.3, 0.3], y(meter): [-0.3, 0.3] and rotation: [-np.pi / 2, np.pi / 2]
#
# if you want to assign a cube position, you can change random_pos_rot to False and use
# "sim_tester.task_env.reset_tar_obj(tar_pos_rot = tar_pos_rot, random_pos_rot = False)" where tar_pos_rot is a 3 element list contains [x, y, rotation]

# Cell
# network.eval()

#断开与仿真环境的连接
if p.isConnected():
    p.disconnect()


import pybullet as p
import pybullet_data
import shutil

#目标检测
import os, sys
import cv2
sys.path.append(os.path.abspath('/home/taizun/embodiedAI/mmdetection/demo'))
sys.path.append(os.path.abspath('/home/taizun/embodiedAI/graspnet-baseline'))
import det_getdata
# import demo_sim_320_256
# from demo_sim_320_256 import PickAndPlace
import demo_sim_3
from demo_sim_3 import PickAndPlace
import argparse
import rospy
import time
from geometry_msgs.msg import Pose
from std_msgs.msg import String

import time  # Import time module
import csv   # Import csv module


#urdf
actions = [
    #  {"function":"semanticSlamNavigation", "paremeters":{"targetID":"C"}}, 
    #  {"function":"semanticSlamNavigation", "paremeters":{"targetID":"B"}}, 
     {"function":"2dDetection", "paremeters":{"object":"apple"}}, 
     {"function":"pick", "paremeters":{"box": [0,0,0,0]}},
    #  {"function":"semanticSlamNavigation", "paremeters":{"site":"C"}}, 
    #  {"function":"2dDetection", "paremeters":{"object":"box"}}, 
    #  {"function":"put", "paremeters":{"box": [0,0,0,0]}},
    #  {"function":"semanticSlamNavigation", "paremeters":{"site":"B"}}, 
     {"function":"end", "paremeters":{}} 
     ]
#初始化目标检测
det = det_getdata.ImageProcess()
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="/home/taizun/embodiedAI/graspnet-baseline/checkpoint-rs.tar",help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()
data_dir = '/home/taizun/embodiedAI/graspnet-baseline/doc/mydata'

res_box = [0,0,0,0]


def quaternion_to_rpy(quaternion):
    r = R.from_quat(quaternion)
    return r.as_euler('xyz', degrees=False)

# def save_image(sim_tester, j,timestamp_str):
#     width = 320
#     height = 256
#     # Determine the folder name based on j
#     folder_suffix = f"data_{j:03d}"
#     folder_rgb = 'rgb'
#     base_path = '/data/yuantingyu/tea_box_9_19'
    
#     # Create folders if they don't exist
#     folder_names = ['topdown', 'front', 'root', 'wrist', 'side','fronttop']
#     for folder_name in folder_names:
#         folder_path = os.path.join(base_path, folder_name, folder_suffix,folder_rgb)
#         os.makedirs(folder_path, exist_ok=True)
    
#     #add camera
#     filename_fronttop = f'{base_path}/fronttop/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
#     viewMatrix_fronttop, projection_matrix_fronttop = sim_tester.get_matrix("fronttop")
#     w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_fronttop, projection_matrix_fronttop, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
#     rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#     cv2.imwrite(filename_fronttop, rgb_image_cv2)

#     # Save topdown image
#     filename_topdown = f'{base_path}/topdown/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
#     viewMatrix_topdown, projection_matrix_topdown = sim_tester.get_matrix("topdown")
#     w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_topdown, projection_matrix_topdown, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
#     rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#     cv2.imwrite(filename_topdown, rgb_image_cv2)

#     # Save front image
#     filename_front = f'{base_path}/front/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
#     viewMatrix_front, projection_matrix_front = sim_tester.get_matrix("front")
#     w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_front, projection_matrix_front, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
#     rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#     cv2.imwrite(filename_front, rgb_image_cv2)

#     # Save root image
#     filename_root = f'{base_path}/root/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
#     viewMatrix_root, projection_matrix_root = sim_tester.get_matrix("root")
#     w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_root, projection_matrix_root, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
#     rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#     cv2.imwrite(filename_root, rgb_image_cv2)

#     # Save wrist image
#     filename_wrist = f'{base_path}/wrist/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
#     viewMatrix_wrist, projection_matrix_wrist = sim_tester.get_matrix("wrist")
#     w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_wrist, projection_matrix_wrist, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
#     rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#     cv2.imwrite(filename_wrist, rgb_image_cv2)

#     # Save side image
#     filename_side = f'{base_path}/side/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
#     viewMatrix_side, projection_matrix_side = sim_tester.get_matrix("side")
#     w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_side, projection_matrix_side, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#     rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
#     rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#     cv2.imwrite(filename_side, rgb_image_cv2)

def save_image(sim_tester, j, timestamp_str, vids):
    width = 320
    height = 256
    # Determine the folder name based on j
    folder_suffix = f"data_{j:03d}"
    folder_rgb = 'rgb'
    base_path = '/data/yuantingyu/apple_9_19'
    
    # Create folders if they don't exist
    folder_names = ['topdown', 'front', 'root', 'wrist', 'side', 'fronttop']
    for folder_name in folder_names:
        folder_path = os.path.join(base_path, folder_name, folder_suffix, folder_rgb)
        os.makedirs(folder_path, exist_ok=True)
    
    # Save fronttop image and add to video list
    filename_fronttop = f'{base_path}/fronttop/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
    viewMatrix_fronttop, projection_matrix_fronttop = sim_tester.get_matrix("fronttop")
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_fronttop, projection_matrix_fronttop, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(filename_fronttop, rgb_image_cv2)
    vids['fronttop'].append(rgb_image_cv2)

    # Save topdown image and add to video list
    filename_topdown = f'{base_path}/topdown/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
    viewMatrix_topdown, projection_matrix_topdown = sim_tester.get_matrix("topdown")
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_topdown, projection_matrix_topdown, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(filename_topdown, rgb_image_cv2)
    vids['topdown'].append(rgb_image_cv2)

    # Save front image and add to video list
    filename_front = f'{base_path}/front/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
    viewMatrix_front, projection_matrix_front = sim_tester.get_matrix("front")
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_front, projection_matrix_front, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(filename_front, rgb_image_cv2)
    vids['front'].append(rgb_image_cv2)

    # Save root image and add to video list
    filename_root = f'{base_path}/root/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
    viewMatrix_root, projection_matrix_root = sim_tester.get_matrix("root")
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_root, projection_matrix_root, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(filename_root, rgb_image_cv2)
    vids['root'].append(rgb_image_cv2)

    # Save wrist image and add to video list
    filename_wrist = f'{base_path}/wrist/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
    viewMatrix_wrist, projection_matrix_wrist = sim_tester.get_matrix("wrist")
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_wrist, projection_matrix_wrist, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(filename_wrist, rgb_image_cv2)
    vids['wrist'].append(rgb_image_cv2)

    # Save side image and add to video list
    filename_side = f'{base_path}/side/{folder_suffix}/rgb/image_{timestamp_str[:6]}.png'
    viewMatrix_side, projection_matrix_side = sim_tester.get_matrix("side")
    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix_side, projection_matrix_side, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(filename_side, rgb_image_cv2)
    vids['side'].append(rgb_image_cv2)

    return vids

#两边均匀采集
def get_random_tar_obj_range():
    ranges = [
        [0.15, 0.4, 0.4, 0.7, -np.pi / 2, np.pi / 2],
        [-0.4, -0.15, 0.4, 0.7, -np.pi / 2, np.pi / 2]
    ]
    return random.choice(ranges)




# 连接到仿真环境并启用GUI
#p.DIRECT  非gui
#p.GUI   使用gui
#设置采集多少次数据
#collect_num = 0
for collect_num in trange(0,800):
    # Initialize a dictionary to store video frames for each view
    vids_all_views = {
        'fronttop': [],
        'topdown': [],
        'front': [],
        'root': [],
        'wrist': [],
        'side': []
    }
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 初始化仿真测试器
    sim_tester = SimTester("pick")
    # tar_obj_range = [-0.3, 0.3, 0.4, 0.7, -np.pi / 2, np.pi / 2]
    # tar_obj_range = [0.15, 0.4, 0.4, 0.7, -np.pi / 2, np.pi / 2]
    tar_obj_range = get_random_tar_obj_range()
    sim_tester.reset_tester(args["cam_view"])
    sim_tester.task_env.reset_tar_obj(tar_obj_range=tar_obj_range, random_pos_rot=True)
    sim_tester.update_imgs(args["cam_view"])
    vids = []


    #为了将夹爪旋转角度跟木块对齐
    tar_obj_pose = p.getBasePositionAndOrientation(sim_tester.task_env.tar_obj)
    # Extract the quaternion from tar_obj_pose
    tar_obj_quaternion = tar_obj_pose[1]
    # Convert the quaternion to Euler angles
    tar_obj_euler = p.getEulerFromQuaternion(tar_obj_quaternion)
    print("木块目标旋转")
    #例子数据：(0.0, 1.5707963267948966, -1.465028214656537)
    print(tar_obj_euler)

    


    #目标检测
    #move banana in env
    #get image and depth
    # width = 1080
    # height = 864
    width = 640
    height = 480
    viewMatrix, projection_matrix = sim_tester.get_matrix("topdown")
    print("viewMatrix")
    print(viewMatrix)
    print("projection_matrix")
    print(projection_matrix)
    q,t=bulletView2cvPose(viewMatrix)
    print("q")
    print(q)
    print("t")
    print(t)

    w, h, rgb, depth, seg = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)


    # Convert the RGB image data to a format suitable for saving
    rgb_image = np.reshape(rgb, (height, width, 4))[:, :, :3]  # Keep only RGB channels
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV


    # Save the RGB image
    cv2.imwrite('rgb_image.png', rgb_image_cv2)

    # Normalize depth image to range [0, 255] and save
    depth_image = np.array(depth).reshape((height, width))

    near = 0.1
    far = 100

    depImg = far * near / (far - (far - near) * depth_image)  # 计算实际深度值
    depImg = np.asarray(depImg).astype(np.float32)* 1000.  # 转换为毫米单位
    # depImg = depImg.astype(np.uint16)  # 转换为16位无符号整数格式

    depth_image_cv2 = (depth_image * 255 / np.max(depth_image)).astype(np.uint8)
    cv2.imwrite('depth_image.png', depth_image_cv2)

    # Convert segmentation mask to a suitable format and save
    seg_image = np.array(seg).reshape((height, width))
    seg_image_cv2 = seg_image.astype(np.uint8)
    cv2.imwrite('seg_image.png', seg_image_cv2)
    print("Images saved successfully.")



    pap = PickAndPlace(data_dir,cfgs)

    for action in actions:
        function = action["function"]

        if function == "2dDetection":
            obj = action["paremeters"]["object"]
            #give image
            res_box = det.get_result(obj)
        elif function == "pick" or function == "put":
            if res_box[0] == 0 and res_box[1] == 0 and res_box[2] == 0 and res_box[3] == 0:
                continue
            box = action["paremeters"]["box"]
            #give image and depth
            res = pap.predict(function, res_box,rgb_image,depImg)
            print("!!!!!!!!!!!!!!!!!!!")
            print(res)
            #res: 3+4(pose)
            #pose->joint
            #calculate delta
            #move and capture



    #后4维是4元数。目标的抓取状态的旋转信息
    tar_quaternion = res[-4:]

    print(f"tar_quaternion转换前: {tar_quaternion}")
    tar_quaternion_rpy = quaternion_to_rpy(tar_quaternion)
    print(f"tar_quaternion_rpy四元数转欧拉角: {tar_quaternion_rpy}")

    #前3维是平移。目标的抓取姿态的位置信息
    tar_translation = res[:3]

    base_position, base_orientation = p.getBasePositionAndOrientation(sim_tester.task_env.robot.robot_id)
    print("base_position")
    print(base_position)
    tar_translation[2] = base_position[2] + tar_translation[2] -0.02 # 因为桌子是0.65和机械臂基坐标系是0.67
    tar_translation[2]=tar_translation[2]-0.05 #人工调整参数




    # terminate_episode = [0]
    # delta_ee_pos = [-0.0725490152835846, -0.04274509474635124, -0.005098029971122742]
    # delta_ee_rot = [0.061599910259246826, 0.120735764503479, -0.42134302854537964]
    # gripper_cmd  = [-0.0039215087890625]
    # sim_tester.test_step(delta_ee_pos, delta_ee_rot, gripper_cmd, args["cam_view"])

    # # 将当前帧添加到视频列表中
    # imgs = sim_tester.imgs
    # vids.append(imgs[-1][0].permute(1, 2, 0).numpy() * 255)

    # terminate_episode = [1]

    terminate_episode = [1]
    gripper_cmd  = [0]
    #获取机器人末端信息
    last_ee_pose = p.getLinkState(
                sim_tester.task_env.robot.robot_id, sim_tester.task_env.robot.ee_index
            )[0:2]

    print("sim_tester.task_env.robot.robot_id")
    print(sim_tester.task_env.robot.robot_id)
    print("sim_tester.task_env.robot.ee_index")
    print(sim_tester.task_env.robot.ee_index)
    print("last_ee_pose")
    print(last_ee_pose)

    #平移的偏移量
    now_translation = list(last_ee_pose[0])
    delta_ee_pos = [t - n for t, n in zip(tar_translation,now_translation)]

    #旋转的偏移量
    # 将 now_quaternion 转换为列表
    now_quaternion =  list(last_ee_pose[1])
    print(f"now_quaternion转换前: {now_quaternion}")
    now_quaternion = quaternion_to_rpy(now_quaternion)
    print(f"now_quaternion四元数转欧拉角: {now_quaternion}")

    #对齐木块的旋转
    tar_quaternion_rpy = [0,0,tar_obj_euler[2]]
    #进行减法运算
    delta_ee_rot = [t - n for t, n in zip(tar_quaternion_rpy,now_quaternion)]

    def split_vector(delta_ee_pos, parts=15, std_dev=0.01):
        # """
        # 将三维度的向量 delta_ee_pos 分成指定数量的部分，每个部分之间有一定的差异，
        # 但是方差不大，并且总和等于 delta_ee_pos。
        
        # :param delta_ee_pos: 一个形状为 (3,) 的 NumPy 数组，表示要分割的向量。
        # :param parts: 整数，表示分割后的部分数量。
        # :param std_dev: 浮点数，表示添加到每一部分的随机噪声的标准差。
        # :return: 形状为 (3, parts) 的 NumPy 数组，每一列代表一个分割的部分。
        # """
        # 初始化结果数组
        
        result = np.zeros((3, parts))
        
        # 将 list 转换成 numpy 数组
        delta_ee_pos_array = np.array(delta_ee_pos)
        # 计算每个维度的平均分配值
        avg_values = delta_ee_pos_array / parts
        
        for i in range(3):
            # 生成随机扰动
            random_perturbations = np.random.normal(loc=avg_values[i], scale=std_dev, size=parts)
            
            # 调整随机数，使它们的总和等于原始值
            adjustment = (delta_ee_pos[i] - np.sum(random_perturbations)) / parts
            adjusted_random_perturbations = random_perturbations + adjustment
            
            # 存储结果
            result[i, :] = adjusted_random_perturbations
        return result

    # delta_ee_pos = [x / 15 for x in delta_ee_pos]
    # delta_ee_rot = [0,0,delta_ee_rot[2]/15]
    delta_ee_rot = [0,0,delta_ee_rot[2]/6]

    print("delta_ee_pos:")
    print(delta_ee_pos)
    print("delta_ee_rot:")
    print(delta_ee_rot)
    delta_ee_pos_vector = split_vector(delta_ee_pos)
    # delta_ee_rot_vector = split_vector(delta_ee_rot)
    print(delta_ee_pos_vector)
    print(np.sum(delta_ee_pos_vector,axis=1))
    # print(np.sum(delta_ee_rot_vector,axis=1))

    

    #时间戳
    start_time = time.time()  # Record the start time
    timestamps = []  # List to store timestamps
    position_diffs = []
    rotation_diffs = []
    gripper_cmds = []

    #delta变换
    ee_abs_pos=[]
    ee_abs_ori=[]
    # 初始化一个空列表用于存储 delta_pose
    delta_poses = []
    # 初始化result_raw数据列表
    data_list = []

    print(f"执行到第{collect_num}组的数据采集")
    for i in trange(32):
        if i >=6:
            delta_ee_rot = [0,0,0]

        if i <15:
            delta_ee_pos = tuple(delta_ee_pos_vector[:, i])
            # delta_ee_rot = tuple(delta_ee_rot_vector[:, i])
        if i >= 15:
            gripper_cmd = [0]
            delta_ee_pos = [0,0,-0.008]
            delta_ee_rot = [0,0,0]
        if i >= 18:
            gripper_cmd = [1]
            delta_ee_pos = [0,0,0.0]
            delta_ee_rot = [0,0,0]
        if i >= 26:
            gripper_cmd = [0]
            delta_ee_pos = [0,0,0.045]
            delta_ee_rot = [0,0,0]

        #获取机器人末端信息
        last_ee_pose_1 = p.getLinkState(
                    sim_tester.task_env.robot.robot_id, sim_tester.task_env.robot.ee_index
                )[0:2]
        now_translation_1 = list(last_ee_pose_1[0])
        now_quaternion_1 =  list(last_ee_pose_1[1])
        now_quaternion_1_rpy = quaternion_to_rpy(now_quaternion_1)

        #这次运动的目标，就是相当于test_step动作执行函数里的cur_ee_pose
        next_translation = np.array(now_translation_1) + np.array(delta_ee_pos)
        next_quaternion = np.array(now_quaternion_1_rpy) + np.array(delta_ee_rot)

        print(f"运动前当前末端位置信息: {now_translation_1}, 目标位置信息: {next_translation}, 位置差: {delta_ee_pos}")
        print(f"运动前当前末端旋转信息: {now_quaternion_1_rpy}, 目标旋转信息: {next_quaternion}, 旋转差: {delta_ee_rot}")

        next_quaternion_4 = p.getQuaternionFromEuler(next_quaternion)

        #join 0~8,result_raw文件中的数据
        tar_j_pos = sim_tester.task_env.robot.calc_ik([next_translation,next_quaternion_4])
        print("tar_j_pos_2")
        print(tar_j_pos)


        # 更新仿真环境
        imgs = sim_tester.imgs
        sim_tester.test_step(delta_ee_pos, delta_ee_rot, gripper_cmd, args["cam_view"])


        #获取运动后机器人末端信息
        last_ee_pose_2 = p.getLinkState(
                    sim_tester.task_env.robot.robot_id, sim_tester.task_env.robot.ee_index
                )[0:2]
        now_translation_2 = list(last_ee_pose_2[0])
        now_quaternion_2 =  list(last_ee_pose_2[1])
        now_quaternion_2_rpy = quaternion_to_rpy(now_quaternion_2)

        print(f"执行一次运动后当前末端位置信息: {now_translation_2}")
        print(f"执行一次运动后当前末端旋转信息: {now_quaternion_2_rpy}")

        if i==0:
            ee_abs_pos.append(now_translation_1)
            ee_abs_ori.append(now_quaternion_1)
        if i>=0:
            ee_abs_pos.append(now_translation_2)
            ee_abs_ori.append(now_quaternion_2)
        


        #记录执行时间戳,除了10.000
        elapsed_time = (time.time() - start_time)/10.000
        timestamps.append(elapsed_time)
        #记录动作执行的delta
        position_diffs.append(np.subtract(now_translation_2, now_translation_1))
        rotation_diffs.append(np.subtract(now_quaternion_2_rpy, now_quaternion_1_rpy))

        # 将数据存储到列表中result_raw文件
        timestamp_list=[elapsed_time]
        data_row = timestamp_list + list(tar_j_pos) + list(tar_translation) + list(tar_quaternion) + gripper_cmd
        data_list.append(data_row)

        #记录夹爪信息
        gripper_cmds.append(gripper_cmd[0])

        # 设置timestamp在图片上的命名
        timestamp_str = f"{elapsed_time:.6f}".replace('.', '')  # Convert to string and remove decimal point
        # Generate the filename based on the formatted timestamp

        #保存图片的集成函数
        vids_all_views =save_image(sim_tester,collect_num,timestamp_str,vids_all_views)


        

        # 将当前帧添加到视频列表中
        imgs = sim_tester.imgs
        vids.append(imgs[-1][0].permute(1, 2, 0).numpy() * 255)

        # 步进仿真以更新GUI显示
        p.stepSimulation()
            
        
    contact_points = p.getContactPoints(bodyA=sim_tester.task_env.robot.robot_id, 
                                    bodyB=sim_tester.task_env.tar_obj)
    
    success = bool(contact_points)

    # 获取目标物体底部的位置
    obj_pos, obj_orientation = p.getBasePositionAndOrientation(sim_tester.task_env.tar_obj)
    table_height = 0.674  # 将此处替换
    some_threshold_value = 0.01
    object_bottom_height = obj_pos[2]  # 物体底部的 z 坐标

    # 检查物体底部是否离开了桌面
    object_lifted = object_bottom_height > table_height + some_threshold_value

    # 更新成功条件
    success = success and object_lifted

    success_message = f"{collect_num}组机械臂成功抓取到了" if success else f"{collect_num}组机械臂未能抓取到"
    print(success_message)

    if success:
        print(f"{collect_num}组机械臂成功抓取到了")
    else:
        print(f"{collect_num}组机械臂未能抓取到")
        
        for folder_name in ['topdown', 'front', 'root', 'wrist', 'side']: 
            folder_path = os.path.join('/data/yuantingyu/apple_9_19', folder_name, f'data_{collect_num:03d}') 
            if os.path.exists(folder_path): 
                shutil.rmtree(folder_path) 
                print(f"Deleted folder: {folder_path}")

    

    time.sleep(0.2)
    # 断开仿真环境
    p.disconnect()

    if contact_points:
        #delta变换
        for i in range(1, len(ee_abs_ori)):
                    inv_last_ee_pose = p.invertTransform(
                        ee_abs_pos[i - 1], ee_abs_ori[i - 1]
                    )
                    delta_pose = p.multiplyTransforms(
                        inv_last_ee_pose[0],
                        inv_last_ee_pose[1],
                        ee_abs_pos[i],
                        ee_abs_ori[i],
                    )
                    # 将 delta_pose 转换为列表
                    delta_pose = list(delta_pose)

                    # 确保 delta_pose[1] 是列表，避免后续修改时报错
                    delta_pose[1] = list(delta_pose[1])

                    # 对 delta_pose[1] 进行修改
                    delta_pose[1] = quaternion_to_rpy(delta_pose[1])

                    # 如果需要，可以将 delta_pose 转回元组
                    delta_pose = tuple(delta_pose)


                    # 将 delta_pose 添加到列表中
                    delta_poses.append(delta_pose)

        print("delta_poses")
        print(delta_poses)

        #设置csv保存的路径
        base_path = f'/data/yuantingyu/apple_9_19'
        folder_name = f'data_{collect_num:03d}'
        front_path = os.path.join(base_path, 'front', folder_name)
        wrist_path = os.path.join(base_path, 'wrist', folder_name)
        topdown_path = os.path.join(base_path, 'topdown', folder_name)
        side_path = os.path.join(base_path, 'side', folder_name)
        root_path = os.path.join(base_path, 'root', folder_name)

        fronttop_path = os.path.join(base_path, 'fronttop', folder_name)
        #创建路径
        os.makedirs(front_path, exist_ok=True)
        os.makedirs(wrist_path, exist_ok=True)
        os.makedirs(topdown_path, exist_ok=True)
        os.makedirs(fronttop_path, exist_ok=True)
        os.makedirs(side_path, exist_ok=True)
        os.makedirs(root_path, exist_ok=True)

        csv_paths = {
            'front': os.path.join(front_path, 'result.csv'),
            'wrist': os.path.join(wrist_path, 'result.csv'),
            'topdown': os.path.join(topdown_path, 'result.csv'),
            'side': os.path.join(side_path, 'result.csv'),
            'root': os.path.join(root_path, 'result.csv'),

            'fronttop': os.path.join(fronttop_path, 'result.csv'),
        }

        for view, csv_path in csv_paths.items():
            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['timestamp', 'ee_command_position_x', 'ee_command_position_y', 'ee_command_position_z',
                                    'ee_command_rotation_x', 'ee_command_rotation_y', 'ee_command_rotation_z', 'gripper_closedness_commanded'])
                for i in range(len(timestamps)):
                    csv_writer.writerow([timestamps[i], 
                                        delta_poses[i][0][0], delta_poses[i][0][1], delta_poses[i][0][2],
                                        delta_poses[i][1][0], delta_poses[i][1][1], delta_poses[i][1][2], gripper_cmds[i]])

        header = [
            'timestamp','joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
            'joint_6', 'joint_7', 'joint_8', 'tar_obj_pose_x', 'tar_obj_pose_y',
            'tar_obj_pose_z', 'tar_obj_pose_rx', 'tar_obj_pose_ry',
            'tar_obj_pose_rz', 'tar_obj_pose_rw', 'gripper_closedness_commanded'
        ]

        raw_csv_paths = {
            'front': os.path.join(front_path, 'result_raw.csv'),
            'wrist': os.path.join(wrist_path, 'result_raw.csv'),
            'topdown': os.path.join(topdown_path, 'result_raw.csv'),
            'side': os.path.join(side_path, 'result_raw.csv'),
            'root': os.path.join(root_path, 'result_raw.csv'),

            'fronttop': os.path.join(fronttop_path, 'result_raw.csv')
        }

        for view, csv_path in raw_csv_paths.items():
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(data_list)

        # # 保存视频
        # from IPython.display import Video
        # vwrite("vis_fangzhen_checkpoint.mp4", vids)
        # Video("vis_fangzhen_checkpoint.mp4", embed=True, width=320, height=512)


        # for view, frames in vids_all_views.items():
        #     output_filename = f'{base_path}/{view}/data_{collect_num:03d}.mp4'
            
        #     # Convert each frame from BGR to RGB
        #     frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            
        #     # Convert frames to the correct format for vwrite (should be a NumPy array)
        #     frames_np = np.array(frames_rgb)
            
        #     # Use vwrite to save the video
        #     vwrite(output_filename, frames_np)
            
        #     print(f"Video saved: {output_filename}")

        for view, frames in vids_all_views.items():
            output_filename = f'{base_path}/{view}/data_{collect_num:03d}.mp4'
            
            # Convert each frame from BGR to RGB and ensure correct resolution
            frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            
            # Convert frames to the correct format for vwrite (NumPy array, and ensure proper resolution)
            frames_np = np.array(frames_rgb)
            
            # Check the resolution and data format (optional)
            height, width, _ = frames_np[0].shape
            print(f"Saving video with resolution {width}x{height}")
            
            # Use vwrite to save the video with higher quality
            vwrite(output_filename, frames_np, outputdict={'-vcodec': 'libx264', '-b': '3000k'})  # Adjust bitrate for quality
            
            print(f"Video saved: {output_filename}")
