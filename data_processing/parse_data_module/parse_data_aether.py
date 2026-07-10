import argparse
import os
import h5py
import cv2
import json
import shutil
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Dict, Union
from common.svo_utils import parse_svo_to_episode
from common.replay_buffer import ReplayBuffer
from scipy.ndimage import uniform_filter1d
from common.interpolation_utils import PoseInterpolator, LinearInterpolator, get_interp1d
from utils_data_process import get_hand_joints_mano_single_hand, MANO_JOINT_CONCAT_ORDER, MANO_JOINT_FAAS_INDEX
from common.pose_utils import pose_to_mat, mat_to_pose, fast_mat_inv, xyz_to_mat, mat_to_xyz


class SequenceTransform_aether:
    """
    A class to store a sequence of transform data from aether Dataset.
    For example, transform data is like (T, 4, 4), where T is the number of frames, 4 is the dimension of the transform matrix.
    """

    def __init__(self, h5_file_dict):

        """
        This h5_file_dict should contains two keys: pose and quaternion.
        """
        self.pose = h5_file_dict['pos'][:]                 # (T, 3)
        self.quaternion = h5_file_dict['quaternion'][:]     # (T, 4)

    def get_pos(self):
        return self.pose   # (T, 3)
    
    def get_quat_orn(self):
        return self.quaternion   # (T, 4)

    def get_rotvec_orn(self):
        return R.from_quat(self.quaternion).as_rotvec()   # (T, 3)
    
    def get_mat_orn(self):
        return R.from_quat(self.quaternion).as_matrix()   # (T, 3, 3)
    
    def get_7d_pose(self):
        return np.concatenate([self.get_pos(), self.get_quat_orn()], axis=-1)   # (T, 7)
    
    def get_6d_pose(self):
        return np.concatenate([self.get_pos(), self.get_rotvec_orn()], axis=-1)   # (T, 6)
    
    def get_mat_pose(self):
        pos = self.get_pos()
        orn = self.get_mat_orn()
        T = pos.shape[0]
        mat_result = np.zeros((T, 4, 4))
        mat_result[:, :3, :3] = orn
        mat_result[:, :3, 3] = pos
        mat_result[:, 3, 3] = 1
        return mat_result                     # (T, 4, 4)
    

class SequenceHandModel_aether:

    """
    A class to store a sequence of hand data from aether Dataset.
    For example, hand pos sequence shape is like (T, 20, 3), where T is the number of frames, 20 is the number of joints, 3 is the dimension of the position.
    """

    def __init__(self, h5_file, is_left=False):
        self.is_left = is_left
        self.joint_ids = [
            'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
        ]
        self.hand_side = 'L' if is_left else 'R'
        pos_list = []
        orn_list = []
        for finger_id in range(5):
            for joint_id in range(4):
                pos_list.append(h5_file['state'][f'FJ_{self.hand_side}_{finger_id}_{joint_id}']['pos'][:])
                orn_list.append(h5_file['state'][f'FJ_{self.hand_side}_{finger_id}_{joint_id}']['quaternion'][:])
        self.pos = np.stack(pos_list, axis=1)
        self.orn = np.stack(orn_list, axis=1)
        self.joint_ids_map = dict()
        for joint_id in range(len(self.joint_ids)):
            self.joint_ids_map[self.joint_ids[joint_id]] = joint_id

    def get_pos(self, joint_id=None):
        if joint_id is None:
            return self.pos   # (T, N, 3)
        else:
            return self.pos[:, self.joint_ids_map[joint_id]]   # (T, 3)

    def get_quat_orn(self, joint_id=None):
        if joint_id is None:
            return self.orn   # (T, N, 4)
        else:
            return self.orn[:, self.joint_ids_map[joint_id]]   # (T, 4)
    
    def get_rotvec_orn(self, joint_id=None):
        if joint_id is None:
            T = self.orn.shape[0]
            return R.from_quat(self.orn.reshape(-1, 4)).as_rotvec().reshape(T, -1, 3)  # (T, N, 3)
        else:
            return R.from_quat(self.orn[:, self.joint_ids_map[joint_id]]).as_rotvec()   # (T, 3)

    def get_mat_orn(self, joint_id=None):
        if joint_id is None:
            T = self.orn.shape[0]
            return R.from_quat(self.orn.reshape(-1, 4)).as_matrix().reshape(T, -1, 3, 3)  # (T, N, 3, 3)
        else:
            return R.from_quat(self.orn[:, self.joint_ids_map[joint_id]]).as_matrix()   # (T, 3, 3)

    def get_7d_pose(self, joint_id=None):
        if joint_id is None:
            return np.concatenate([self.get_pos(), self.get_quat_orn()], axis=-1)   # (T, N, 7)
        else:
            return np.concatenate([self.get_pos(joint_id), self.get_quat_orn(joint_id)], axis=-1)   # (T, 7)

    def get_6d_pose(self, joint_id=None):
        if joint_id is None:
            return np.concatenate([self.get_pos(), self.get_rotvec_orn()], axis=-1)   # (T, N, 6)
        else:
            return np.concatenate([self.get_pos(joint_id), self.get_rotvec_orn(joint_id)], axis=-1)   # (T, 6)

    def get_mat_pose(self, joint_id=None):
        pos = self.get_pos(joint_id)
        orn = self.get_mat_orn(joint_id)
        if joint_id is None:
            T, N = pos.shape[:2]
            mat_result = np.zeros((T, N, 4, 4))   # (T, N, 4, 4)
            mat_result[:, :, :3, :3] = orn
            mat_result[:, :, :3, 3] = pos
            mat_result[:, :, 3, 3] = 1
        else:
            T = pos.shape[0]
            mat_result = np.zeros((T, 4, 4))   # (T, 4, 4)
            mat_result[:, :3, :3] = orn
            mat_result[:, :3, 3] = pos
        return mat_result


def filter_tactile_data(data_ori: np.ndarray, baseline_frames=10, noise_threshold=0.2, use_moving_avg=True, window_size=3):
    """
    input:
        data_ori: np.ndarray, shape (T, N, D), 原始触觉数据
        baseline_frames: int, 用于基线校正的前N帧
        noise_threshold: float, 噪声阈值，小于该值的数据将被设为0
        use_moving_avg: bool, 是否使用移动平均滤波
        window_size: int, 移动平均滤波的窗口大小
    output:
        data_filtered: np.ndarray, shape (T, N, D), 过滤后的触觉数据
    """
    
    data_filtered = data_ori.copy()
    
    # 1. 基线校正：减去前N帧的均值
    baseline = np.mean(data_filtered[:baseline_frames], axis=0)
    data_filtered = data_filtered - baseline
    
    # 2. 阈值过滤：将小于阈值的值设为0（去除噪声）
    data_filtered = np.where(data_filtered < noise_threshold, 0, data_filtered)
    
    # 3. 移动平均滤波：平滑数据
    if use_moving_avg and window_size > 1:
        data_filtered = uniform_filter1d(data_filtered, size=window_size, axis=0, mode='nearest')
    
    # 确保没有负值
    data_filtered = np.clip(data_filtered, 0, None)
    
    return data_filtered


def get_tactile_from_h5file(
    h5_file,
    type='fingers',
    is_left=True,
    is_filtered=True,
    noise_threshold: float = 0.2,
    baseline_frames: int = 10,
    use_moving_avg: bool = True,
    window_size: int = 3,
):
    """
    type: 'fingers' or 'palm'
    """
    assert type in ['fingers', 'palm'], "type must be 'fingers' or 'palm'"
    tactile_data_raw = h5_file['state']['LeftPort'] if is_left else h5_file['state']['RightPort']
    if type == 'fingers':
        # from thumb to pinky
        finger_id = [list(range(i * 12 + 1, (i + 1) * 12 + 1)) for i in range(5)]
        finger_id.reverse() if is_left else None
        tactile_data = []
        for finger in finger_id:
            finger_data = np.concatenate([tactile_data_raw[f'data_{fid}'][:].reshape(-1, 1) for fid in finger], axis=-1)  # (T, 12)
            tactile_data.append(finger_data)
        tactile_data = np.stack(tactile_data, axis=1)  # (T, 5, 12)
    else:
        tactile_data = np.stack([tactile_data_raw[f'data_{id}'][:].reshape(-1, 1) for id in range(66, 138)], axis=-1)  # (T, 1, 72)

    # print('Use Simple Filter for Tactile Data Processing ...')
    tactile_data_filtered = (
        filter_tactile_data(
            tactile_data,
            baseline_frames=baseline_frames,
            noise_threshold=float(noise_threshold),
            use_moving_avg=use_moving_avg,
            window_size=window_size,
        )
        if is_filtered
        else tactile_data
    )  # (T, N, D)

    # # mse difference before and after filtering
    # mse_diff = np.mean((tactile_data_filtered - tactile_data) ** 2)
    # print(f"Tactile Data MSE Difference before and after filtering: {mse_diff}")
    
    return tactile_data_filtered


def bytes_to_int_numpy(bytes_array, dtype='int32', byteorder='little'):
    """
    使用 numpy.frombuffer 将 bytes 转换为整数数组
    
    参数:
        bytes_array: bytes 或 bytearray 对象
        dtype: 目标数据类型 ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')
        byteorder: 字节序 ('little' 或 'big')
    
    返回:
        numpy 整数数组
    """
    # 方法 1a: 直接使用 frombuffer
    arr = np.frombuffer(bytes_array, dtype=dtype)
    
    # 如果需要指定字节序
    if byteorder == 'big':
        arr = arr.byteswap().newbyteorder()
    
    return arr


def parse_aether_h5file_to_episode(
    h5_file,
    speed_downsample_ratio=1.0,
    fill_instruction_length=100,
    tactile_noise_threshold_fingers: float = 0.2,
    tactile_noise_threshold_palm: float = 0.2,
):

    episode = dict()
    left_hand_aether = SequenceHandModel_aether(h5_file, is_left=True)
    right_hand_aether = SequenceHandModel_aether(h5_file, is_left=False)
    head_tracker_aether = SequenceTransform_aether(h5_file['state']['HeadTracker'])
    left_wrist_aether = SequenceTransform_aether(h5_file['state']['LeftHand'])
    right_wrist_aether = SequenceTransform_aether(h5_file['state']['RightHand'])
    timestamps_org = h5_file['TimeStamps'][:].reshape(-1)

    episode['sub_task_instruction'] = h5_file['Sub_Task_Instruction'][:].astype(str)
    T = episode['sub_task_instruction'].shape[0]
    left_usage = np.asarray(h5_file['Left_Hand_Usage'][:]).reshape(-1)
    right_usage = np.asarray(h5_file['Right_Hand_Usage'][:]).reshape(-1)
    # Aether usage labels occasionally contain NaN; treat invalid values as empty hand.
    left_usage = np.nan_to_num(left_usage.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    right_usage = np.nan_to_num(right_usage.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    episode['left_hand_usage'] = (left_usage > 0.5).astype(np.uint8)
    episode['right_hand_usage'] = (right_usage > 0.5).astype(np.uint8)

    episode['left_hand_pose_aether_dataset'] = left_hand_aether.get_6d_pose()
    episode['right_hand_pose_aether_dataset'] = right_hand_aether.get_6d_pose()
    episode['head_tracker_pose'] = head_tracker_aether.get_6d_pose()
    episode['left_wrist_pose'] = left_wrist_aether.get_6d_pose()
    episode['right_wrist_pose'] = right_wrist_aether.get_6d_pose()

    left_tactile_data_fingers = get_tactile_from_h5file(
        h5_file,
        type='fingers',
        is_left=True,
        noise_threshold=tactile_noise_threshold_fingers,
    )
    right_tactile_data_fingers = get_tactile_from_h5file(
        h5_file,
        type='fingers',
        is_left=False,
        noise_threshold=tactile_noise_threshold_fingers,
    )
    episode['left_tactile_data_fingers'] = left_tactile_data_fingers.reshape(T, 5, 12)
    episode['right_tactile_data_fingers'] = right_tactile_data_fingers.reshape(T, 5, 12)
    episode['left_tactile_area_fingers'] = np.stack([np.array([0, 1, 2, 3, 4])]*T, axis=0)
    episode['right_tactile_area_fingers'] = np.stack([np.array([0, 1, 2, 3, 4])]*T, axis=0)
    episode['left_tactile_sensor_fingers'] = np.stack(['AetherGloveV1']*T, axis=0)
    episode['right_tactile_sensor_fingers'] = np.stack(['AetherGloveV1']*T, axis=0)
    episode['left_tactile_type_fingers'] = np.stack(['matrix']*T, axis=0)
    episode['right_tactile_type_fingers'] = np.stack(['matrix']*T, axis=0)
    
    left_tactile_data_palm = get_tactile_from_h5file(
        h5_file,
        type='palm',
        is_left=True,
        noise_threshold=tactile_noise_threshold_palm,
    )
    right_tactile_data_palm = get_tactile_from_h5file(
        h5_file,
        type='palm',
        is_left=False,
        noise_threshold=tactile_noise_threshold_palm,
    )
    left_palm_1 = left_tactile_data_palm[:, :, :12]
    left_palm_2 = left_tactile_data_palm[:, :, 12:].reshape(T, 1, 4, 15)
    right_palm_1 = right_tactile_data_palm[:, :, :12]
    right_palm_2 = right_tactile_data_palm[:, :, 12:].reshape(T, 1, 4, 15)
    left_palm_1 = np.concatenate([left_palm_1.reshape(T, 1, 1, 12), np.zeros((T, 1, 1, 3))], axis=-1)
    right_palm_1 = np.concatenate([np.zeros((T, 1, 1, 3)), right_palm_1.reshape(T, 1, 1, 12)], axis=-1)
    left_palm_tactile = np.concatenate([left_palm_1, left_palm_2], axis=-2).reshape(T, 1, 75)
    right_palm_tactile = np.concatenate([right_palm_1, right_palm_2], axis=-2).reshape(T, 1, 75)
    
    episode['left_tactile_data_palm'] = left_palm_tactile
    episode['right_tactile_data_palm'] = right_palm_tactile
    episode['left_tactile_area_palm'] = np.stack([np.array([5])]*T, axis=0)
    episode['right_tactile_area_palm'] = np.stack([np.array([5])]*T, axis=0)
    episode['left_tactile_sensor_palm'] = np.stack(['AetherGloveV1']*T, axis=0)
    episode['right_tactile_sensor_palm'] = np.stack(['AetherGloveV1']*T, axis=0)
    episode['left_tactile_type_palm'] = np.stack(['matrix']*T, axis=0)
    episode['right_tactile_type_palm'] = np.stack(['matrix']*T, axis=0)

    # Downsample the Data with Interpolation
    dt_org = (timestamps_org[1:] - timestamps_org[:-1]).mean()
    dt = dt_org / speed_downsample_ratio
    T_start = timestamps_org[0]
    T_end = timestamps_org[-1]
    n_steps = int((T_end - T_start) / dt)
    timestamps = np.arange(n_steps + 1) * dt + T_start
    
    # Debug output to verify speed_downsample_ratio is being used
    if speed_downsample_ratio != 1.0:
        print(f"[Aether] speed_downsample_ratio={speed_downsample_ratio:.3f}: "
              f"original frames={len(timestamps_org)}, resampled frames={len(timestamps)}, "
              f"dt_org={dt_org:.6f}s, dt_new={dt:.6f}s")

    index_interpolate_keys = ['sub_task_instruction', 
                              'left_hand_usage', 
                              'right_hand_usage']
    
    index_interpolator = get_interp1d(timestamps_org, np.arange(T))
    index_idxs = index_interpolator(timestamps)
    index_idxs = index_idxs.astype(int)
    for key in index_interpolate_keys:
        episode[key] = episode[key][index_idxs]
    for key in episode.keys():
        if key in index_interpolate_keys:
            continue
        if 'tactile_sensor' in key or 'tactile_type' in key:
            continue
        value = episode[key]
        last_dim = value.shape[-1]
        # assert value.shape[-1] == 6, f"The last dimension of {key} should be 6 as pos-rotvec vector."
        if value.ndim > 2:
            assert value.ndim == 3, f"The dimension of {key} should be 3 as (T, N, D)."
            n_item = value.shape[1]
        else:
            value = value.reshape(-1, 1, last_dim)
            n_item = 1

        interpolated_value = np.zeros((timestamps.shape[0], n_item, last_dim), dtype=value.dtype)
        for i in range(n_item):
            value_item = value[:, i, :]
            if 'tactile' in key:
                if 'tactile_area' in key:
                    value_item = np.stack([value_item[0]]*timestamps.shape[0], axis=0)
                elif 'tactile_data' in key:
                    value_interpolator = get_interp1d(timestamps_org, value_item)
                    value_item = value_interpolator(timestamps)
                else:
                    raise ValueError(f"Unknown tactile key: {key}, should has 'tactile_area' or 'tactile_data' in the key.")
            elif 'pose' in key:
                value_interpolator = PoseInterpolator(t=timestamps_org, x=value_item) if not key.endswith('_tactile') else LinearInterpolator(t=timestamps_org, x=value_item)
                value_item = value_interpolator(timestamps)
            else:
                raise ValueError(f"Unknown key: {key}, should has 'tactile' or 'pose' in the key.")
            interpolated_value[:, i, :] = value_item
        if n_item == 1:
            if 'pose' in key or 'tactile_area' in key:
                interpolated_value = interpolated_value.reshape(-1, last_dim)
            elif 'tactile_data' in key or 'tactile_sensor' in key:
                pass 
            else:
                raise ValueError(f"Unknown key: {key}, should has 'pose' or 'tactile' in the key.")
        episode[key] = interpolated_value
    
    for key in episode.keys():
        if 'tactile_sensor' in key:
            episode[key] = np.array([episode[key][0]]*len(timestamps))
        if 'tactile_type' in key:
            episode[key] = np.array([episode[key][0]]*len(timestamps))

    episode['timestamps'] = timestamps
    # Use actual length after interpolation, not original T
    T_new = len(timestamps)
    episode['left_tactile_data_fingers'] = episode['left_tactile_data_fingers'].reshape(T_new, 5, 4, 3)
    episode['right_tactile_data_fingers'] = episode['right_tactile_data_fingers'].reshape(T_new, 5, 4, 3)
    episode['left_tactile_data_palm'] = episode['left_tactile_data_palm'].reshape(T_new, 1, 5, 15)
    episode['right_tactile_data_palm'] = episode['right_tactile_data_palm'].reshape(T_new, 1, 5, 15)
    instruction = episode['sub_task_instruction']
    instruction_real = list()
    for ins in instruction:
        instruction_real.append(ins.ljust(fill_instruction_length))
    episode['sub_task_instruction'] = np.array(instruction_real)

    # for key in episode.keys():
    #     print(f"{key}: {episode[key].shape}")
    # import pdb; pdb.set_trace()
    
    return episode


def fix_aether_geometry_coordinate(episode):
    camera_poses = episode['head_tracker_pose']        # (T, 6), rotvec
    left_wrist_pose = episode['left_wrist_pose']
    right_wrist_pose = episode['right_wrist_pose']
    left_finger_pose = episode['left_hand_pose_aether_dataset']
    right_finger_pose = episode['right_hand_pose_aether_dataset']

    ################ Update Camera Coordinate Definition #################
    T = camera_poses.shape[0]
    vive2camera = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    camera_poses = mat_to_pose(pose_to_mat(camera_poses) @ vive2camera)

    ################ Update Wrist & Hand Tracker Pose (Relative to Camera) ###########
    camera_poses = pose_to_mat(camera_poses)
    camera_poses_inv = np.linalg.inv(camera_poses)
    left_wrist_pose = pose_to_mat(left_wrist_pose)
    right_wrist_pose = pose_to_mat(right_wrist_pose)
    left_finger_pose = pose_to_mat(left_finger_pose.reshape(-1, 6)).reshape(T, -1, 4, 4)
    right_finger_pose = pose_to_mat(right_finger_pose.reshape(-1, 6)).reshape(T, -1, 4, 4)
    left_wrist_pose = camera_poses_inv @ left_wrist_pose
    right_wrist_pose = camera_poses_inv @ right_wrist_pose
    left_finger_pose = camera_poses_inv[:, None] @ left_finger_pose
    right_finger_pose = camera_poses_inv[:, None] @ right_finger_pose

    Rz180 = np.eye(4)
    Rz180[:3, :3] = R.from_euler("z", np.pi).as_matrix()
    reflect_x = np.array([-1, 1, 1, 1, -1, -1])
    tracker2camera_bias_left = np.array([0, -0.08, -0.01, 0, 0, 0])
    tracker2camera_bias_right = np.array([-0.05, -0.07, -0.04, 0, 0, 0])
    left_wrist_pose = pose_to_mat(mat_to_pose(Rz180 @ left_wrist_pose) * reflect_x + tracker2camera_bias_left)
    right_wrist_pose = pose_to_mat(mat_to_pose(Rz180 @ right_wrist_pose) * reflect_x + tracker2camera_bias_right)
    left_finger_pose = pose_to_mat(mat_to_pose(Rz180 @ left_finger_pose.reshape(-1, 4, 4)) * reflect_x + tracker2camera_bias_left).reshape(T, -1, 4, 4)
    right_finger_pose = pose_to_mat(mat_to_pose(Rz180 @ right_finger_pose.reshape(-1, 4, 4)) * reflect_x + tracker2camera_bias_right).reshape(T, -1, 4, 4)

    tracker_fix_bias = np.array([0.08, 0.0, 0.0])
    left_wrist_pose[:, :3, -1] = left_wrist_pose[:, :3, -1] + tracker_fix_bias
    right_wrist_pose[:, :3, -1] = right_wrist_pose[:, :3, -1] + tracker_fix_bias

    left_wrist_pose = camera_poses @ left_wrist_pose
    right_wrist_pose = camera_poses @ right_wrist_pose
    left_finger_pose = camera_poses[:, None] @ left_finger_pose
    right_finger_pose = camera_poses[:, None] @ right_finger_pose
    left_wrist_pose = mat_to_pose(left_wrist_pose)
    right_wrist_pose = mat_to_pose(right_wrist_pose)
    left_finger_pose = mat_to_pose(left_finger_pose.reshape(-1, 4, 4)).reshape(T, -1, 6)
    right_finger_pose = mat_to_pose(right_finger_pose.reshape(-1, 4, 4)).reshape(T, -1, 6)
    camera_poses = mat_to_pose(camera_poses)

    ################ Update Wrist Tracker Coordinate Definition #################
    lefttracker2standard = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    righttracker2standard = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    left_wrist_pose = mat_to_pose(pose_to_mat(left_wrist_pose) @ lefttracker2standard)
    right_wrist_pose = mat_to_pose(pose_to_mat(right_wrist_pose) @ righttracker2standard)

    ######################### Finish  ################################

    episode['camera_ego_pose'] = camera_poses       # (T, 6), world coordinate, rotvec
    episode['left_wrist_pose'] = left_wrist_pose    # (T, 6)
    episode['right_wrist_pose'] = right_wrist_pose  # (T, 6)
    episode['left_hand_pose'] = left_finger_pose    # (T, 21, 6)
    episode['right_hand_pose'] = right_finger_pose  # (T, 21, 6)

    del episode['head_tracker_pose']
    del episode['left_hand_pose_aether_dataset']
    del episode['right_hand_pose_aether_dataset']

    # get left-hand keypoints based on wrist coordidnate
    
    return episode


def get_hands_joints_mano_hand(episode):
    assert 'left_hand_pose' in episode.keys() or 'right_hand_pose' in episode.keys(), "Left and right hand pose are not found in the episode."
    assert episode['left_hand_pose'].shape[-1] == 6
    if 'left_hand_pose' in episode.keys():
        left_hand_pose = episode['left_hand_pose']
        left_wrist_pose = episode['left_wrist_pose']
        left_hand_joints, left_hand_keypoints = get_hand_joints_mano_single_hand(left_hand_pose, left_wrist_pose, is_left=True, return_concat_joints=True, mano_joint_concat_order=MANO_JOINT_CONCAT_ORDER)
        episode['left_hand_keypoints'] = left_hand_keypoints    # (T, 21)
        episode['left_hand_joints'] = left_hand_joints          # (T，21)
        T = left_hand_joints.shape[0]
        episode['left_hand_joints_idx'] = np.array(MANO_JOINT_FAAS_INDEX)[None, :].repeat(T, axis=0)
    if 'right_hand_pose' in episode.keys():
        right_hand_pose = episode['right_hand_pose']
        right_wrist_pose = episode['right_wrist_pose']
        right_hand_joints, right_hand_keypoints = get_hand_joints_mano_single_hand(right_hand_pose, right_wrist_pose, is_left=False, return_concat_joints=True, mano_joint_concat_order=MANO_JOINT_CONCAT_ORDER)
        episode['right_hand_keypoints'] = right_hand_keypoints   # (T, 21)
        episode['right_hand_joints'] = right_hand_joints         # (T, 21)
        T = right_hand_joints.shape[0]
        episode['right_hand_joints_idx'] = np.array(MANO_JOINT_FAAS_INDEX)[None, :].repeat(T, axis=0)
    return episode


def parse_aether_data(base_dir, 
                     data_id, 
                     mode,
                     speed_downsample_ratio=1.0,
                     tactile_noise_threshold_fingers: float = 0.2,
                     tactile_noise_threshold_palm: float = 0.2,
                     out_resolutions_resize: Union[None, tuple]=None,      # (width, height)
                     out_resolutions_crop: Union[None, tuple]=None,  
                     out_resolutions_crop_random_ratio: Union[None, tuple]=None,      # (width, height)
                     out_resolutions_image_final: Union[None, tuple]=None, # (width, height)
                     ):

    h5_filepath = os.path.join(base_dir, data_id, f"Data_{data_id}.hdf5")
    h5_file = h5py.File(h5_filepath, "r")
    episode = parse_aether_h5file_to_episode(
        h5_file,
        speed_downsample_ratio=speed_downsample_ratio,
        tactile_noise_threshold_fingers=tactile_noise_threshold_fingers,
        tactile_noise_threshold_palm=tactile_noise_threshold_palm,
    )
    dt = (episode['timestamps'][1:] - episode['timestamps'][:-1]).mean()
    svo_filepath = os.path.join(base_dir, data_id, f"Image_{data_id}.svo2")
    json_filepath = os.path.join(base_dir, data_id, f"Camera_{data_id}.json")
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)
    serial_id = json_data['zedParams']['cameraSerialNumber']
    # T_start = 0
    # T_end = 75
    # for key in episode.keys():
    #     episode[key] = episode[key][T_start:T_end]
    episode = parse_svo_to_episode(svo_filepath, serial_id, episode, mode, dt, camera_name='camera_ego', 
                                   out_resolutions_resize=out_resolutions_resize, 
                                   out_resolutions_crop=out_resolutions_crop, 
                                   out_resolutions_crop_random_ratio=out_resolutions_crop_random_ratio, 
                                   out_resolutions_image_final=out_resolutions_image_final)
    episode = fix_aether_geometry_coordinate(episode)
    episode = get_hands_joints_mano_hand(episode)

    episode_save = dict()

    use_key = [
        'timestamps',
        'camera_ego_pose',
        'camera_ego_rgb',
        'camera_ego_real_timestamp',
        'left_wrist_pose',
        'right_wrist_pose',
        'left_hand_usage',
        'right_hand_usage',
        'left_hand_joints',
        'right_hand_joints',
        'left_hand_joints_idx',
        'right_hand_joints_idx',
        'left_tactile_data_fingers', 'left_tactile_area_fingers', 'left_tactile_sensor_fingers', 'left_tactile_type_fingers',
        'right_tactile_data_fingers', 'right_tactile_area_fingers', 'right_tactile_sensor_fingers', 'right_tactile_type_fingers',
        'left_tactile_data_palm', 'left_tactile_area_palm', 'left_tactile_sensor_palm', 'left_tactile_type_palm',
        'right_tactile_data_palm', 'right_tactile_area_palm', 'right_tactile_sensor_palm', 'right_tactile_type_palm',
        'sub_task_instruction'
    ]

    for key in use_key:
        episode_save[key] = episode[key]
    
    for key in episode_save.keys():
        print(f"{key}: {episode_save[key].shape}")
    # import pdb; pdb.set_trace()
    return episode_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--speed_downsample_ratio", type=float, default=1.0)
    parser.add_argument("--bin_extend_frames", type=int, default=64)
    parser.add_argument("--mode", type=str, default='o', choices=['o', 'p', 's', 'd', 'a'])
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = 'output'

    file_list = os.listdir(args.base_dir)
    folder_list = [file for file in file_list if os.path.isdir(os.path.join(args.base_dir, file))]
    
    # Initialize batch tracking: 50 trajectories per zarr file
    trajectories_per_zarr = 100
    current_zarr_idx = 0
    trajectory_count_in_current_zarr = 0
    replay_buffer = None
    zarr_file_aether = None
    
    def get_or_create_zarr_file(zarr_idx):
        """Get or create a new zarr file for the given index."""
        zarr_path = os.path.join(args.save_dir, f'episode_batch_{zarr_idx:04d}.zarr')
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        return ReplayBuffer.create_from_path(zarr_path, mode='a'), zarr_path
    
    for idx, data_id in enumerate(folder_list):
        
        
        # Create new zarr file if needed (first iteration or reached 50 trajectories)
        if replay_buffer is None or trajectory_count_in_current_zarr >= trajectories_per_zarr:
            if replay_buffer is not None:
                print(f"Completed zarr file {current_zarr_idx:04d} with {trajectory_count_in_current_zarr} trajectories")
            replay_buffer, zarr_file_aether = get_or_create_zarr_file(current_zarr_idx)
            trajectory_count_in_current_zarr = 0
            current_zarr_idx += 1
            print(f"Created new zarr file: {zarr_file_aether}")
        
        try:
            episode = parse_aether_data(args.base_dir, data_id, 
                                        mode=args.mode,
                                        speed_downsample_ratio=args.speed_downsample_ratio,
                                        out_resolutions_resize=(1280, 720),
                                        out_resolutions_crop=(1024, 680),
                                        out_resolutions_crop_random_ratio=(1.0, 1.0),
                                        out_resolutions_image_final=(224, 224))
            # import pdb; pdb.set_trace()
        except Exception as e:
            print(f"Error parsing data for {data_id}: {e}")
            continue

        length = len(episode['timestamps'])
        episode_list = []
        if args.max_frames is not None:
            n_split = (length - args.bin_extend_frames) // args.max_frames
            for i in range(n_split):
                episode_i = {k: v[i*args.max_frames:(i+1)*args.max_frames+args.bin_extend_frames] for k, v in episode.items()}
                episode_list.append(episode_i)
                replay_buffer.add_episode(episode_i, compressors='disk')
                trajectory_count_in_current_zarr += 1
                
                # Check if we need a new zarr file
                if trajectory_count_in_current_zarr >= trajectories_per_zarr:
                    print(f"Completed zarr file {current_zarr_idx-1:04d} with {trajectory_count_in_current_zarr} trajectories")
                    replay_buffer, zarr_file_aether = get_or_create_zarr_file(current_zarr_idx)
                    trajectory_count_in_current_zarr = 0
                    current_zarr_idx += 1
                    print(f"Created new zarr file: {zarr_file_aether}")
            
            if length % args.max_frames != 0:
                episode_i = {k: v[n_split*args.max_frames:(n_split+1)*args.max_frames+args.bin_extend_frames] for k, v in episode.items()}
                episode_list.append(episode_i)
                replay_buffer.add_episode(episode_i, compressors='disk')
                trajectory_count_in_current_zarr += 1
                
                # Check if we need a new zarr file
                if trajectory_count_in_current_zarr >= trajectories_per_zarr:
                    print(f"Completed zarr file {current_zarr_idx-1:04d} with {trajectory_count_in_current_zarr} trajectories")
                    replay_buffer, zarr_file_aether = get_or_create_zarr_file(current_zarr_idx)
                    trajectory_count_in_current_zarr = 0
                    current_zarr_idx += 1
                    print(f"Created new zarr file: {zarr_file_aether}")
        else:
            replay_buffer.add_episode(episode, compressors='disk')
            trajectory_count_in_current_zarr += 1
            
            # Check if we need a new zarr file
            if trajectory_count_in_current_zarr >= trajectories_per_zarr:
                print(f"Completed zarr file {current_zarr_idx-1:04d} with {trajectory_count_in_current_zarr} trajectories")
                replay_buffer, zarr_file_aether = get_or_create_zarr_file(current_zarr_idx)
                trajectory_count_in_current_zarr = 0
                current_zarr_idx += 1
                print(f"Created new zarr file: {zarr_file_aether}")
    
    # Print final zarr file info
    if replay_buffer is not None:
        print(f"Completed final zarr file {current_zarr_idx-1:04d} with {trajectory_count_in_current_zarr} trajectories")
