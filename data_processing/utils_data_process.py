import argparse
import os
import h5py
import cv2
import json
import shutil
from scipy.spatial.transform import Rotation as R
import numpy as np
from common.pose_utils import pose_to_mat, mat_to_pose, fast_mat_inv, xyz_to_mat, mat_to_xyz


MANO_JOINT_CONCAT_ORDER = [
    'palm_adduction', 'thumb_adduction', 'thumb_mcp', 'thumb_ip',
    'index_adduction', 'index_mcp', 'index_pip', 'index_dip',
    'middle_adduction', 'middle_mcp', 'middle_pip', 'middle_dip',
    'ring_adduction', 'ring_mcp', 'ring_pip', 'ring_dip',
    'pinky_adduction', 'pinky_mcp', 'pinky_pip', 'pinky_dip',
    'palm',
]

MANO_JOINT_FAAS_INDEX = [
    1, 26, 2, 3,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24,
    27
]


def calculate_signed_angle_between_vectors(v1, v2, normal, direct_define_v2=False, debug=False):
    """
    计算两个向量之间的有符号角度
    
    Args:
        v1: (..., 3) 或 (3,) 向量1
        v2: (..., 3) 或 (3,) 向量2
        normal: (..., 3) 或 (3,) 法向量，用于确定符号
        direct_define_v2: 是否交换v1和v2
        
    Returns:
        (...,) 或 标量：有符号角度（弧度）
    """
    if direct_define_v2 is True:
        v3 = v2
        v2 = v1
        v1 = v3
    
    # np.cross 支持batch维度
    cross = np.cross(v1, v2)  # (..., 3)
    
    # 处理batch维度的点积
    if cross.ndim > 1:
        sign = np.sign(np.einsum('...i,...i->...', cross, normal))
        dot_v1_v2 = np.einsum('...i,...i->...', v1, v2)
        v1_norm = np.linalg.norm(v1, axis=-1)
        v2_norm = np.linalg.norm(v2, axis=-1)
    else:
        sign = np.sign(np.dot(cross, normal))
        dot_v1_v2 = np.dot(v1, v2)
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
    
    # 避免除以零和数值不稳定
    denominator = v1_norm * v2_norm
    cos_angle = np.clip(dot_v1_v2 / (denominator + 1e-8), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    if debug is True:
        import pdb; pdb.set_trace()
    
    return angle * sign


def calculate_signed_angle_between_planes(v1, v2, v3, normal, direct_define_v2=False):
    """
    计算两个平面之间的有符号角度
    
    Args:
        v1: (..., 3) 或 (3,) 向量1
        v2: (..., 3) 或 (3,) 向量2，定义平面
        v3: (..., 3) 或 (3,) 向量3，定义平面
        normal: (..., 3) 或 (3,) 法向量，用于确定符号
        direct_define_v2: 是否交换v1和v2
        
    Returns:
        (...,) 或 标量：有符号角度（弧度）
    """
    # first define the plane based on v2, v3 and (0,0,0), then project v1 to the plane
    # then calculate signed angle between projected v1 and v2
    # v2 is the base to 
    plane_normal = np.cross(v2, v3)  # (..., 3)
    
    # 处理batch维度的范数和归一化
    if plane_normal.ndim > 1:
        plane_norm = np.linalg.norm(plane_normal, axis=-1, keepdims=True)
        plane_normal = plane_normal / (plane_norm + 1e-8)
        # 投影v1到平面
        dot_v1_plane = np.einsum('...i,...i->...', v1, plane_normal)
        if dot_v1_plane.ndim < plane_normal.ndim:
            dot_v1_plane = dot_v1_plane[..., None]
        projected_v1 = v1 - dot_v1_plane * plane_normal
    else:
        plane_norm = np.linalg.norm(plane_normal)
        plane_normal = plane_normal / (plane_norm + 1e-8)
        dot_v1_plane = np.dot(v1, plane_normal)
        projected_v1 = v1 - dot_v1_plane * plane_normal
    
    return calculate_signed_angle_between_vectors(projected_v1, v2, normal, direct_define_v2)


def get_hand_joints_mano_single_hand(hand_pose, wrist_pose, 
                                     is_left=False, 
                                     return_concat_joints=False,
                                     mano_joint_concat_order=None,):
    
    """
    hand_pose order:
        MANO Hands 21 Points Order:
        self.joint_ids = [
            'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
        ]
    """

    assert hand_pose.shape[-1] == 6 and wrist_pose.shape[-1] == 6, "Hand and wrist pose should be 6D pose."
    
    # Calculate the hand keypoints in wrist coordinate
    T = hand_pose.shape[0]
    hand_pose_mat = pose_to_mat(hand_pose.reshape(-1, 6)).reshape(T, -1, 4, 4)
    wrist_pose_mat = pose_to_mat(wrist_pose)
    hand_pose_mat = np.concatenate([wrist_pose_mat[:, None], hand_pose_mat], axis=1)
    hand_pose_mat = np.linalg.inv(wrist_pose_mat)[:,None] @ hand_pose_mat   # (Wrist->World)^{-1} * (Hand->World)
    hand_pose_mat = mat_to_pose(hand_pose_mat.reshape(-1, 4, 4)).reshape(T, -1, 6)
    kps = hand_pose_mat[..., :3]   # hand keypoints (21, 3) in wrist coordinate 
    
    if is_left is False:

        # Calculate the right hand joints in wrist coordinate, need to use relative action during pretraining
        palm_in_norm = np.cross(kps[:,5] - kps[:,0], kps[:,13] - kps[:,0])

        joint_thumb_mcp = calculate_signed_angle_between_vectors(kps[:,2] - kps[:,1], kps[:,3] - kps[:,2], kps[:,2] - kps[:,5])
        # For mcp, we replace it with the angle between tip & palm to adopt better to 6DoF robot hand  (the change of angle from neighbour joint is to small)
        joint_index_mcp = calculate_signed_angle_between_vectors(kps[:,5] - kps[:,0], kps[:,8] - kps[:,5], kps[:,5] - kps[:,9])
        joint_middle_mcp = calculate_signed_angle_between_vectors(kps[:,9] - kps[:,0], kps[:,12] - kps[:,9], kps[:,9] - kps[:,13])
        joint_ring_mcp = calculate_signed_angle_between_vectors(kps[:,13] - kps[:,0], kps[:,16] - kps[:,13], kps[:,13] - kps[:,17])
        joint_pinky_mcp = calculate_signed_angle_between_vectors(kps[:,17] - kps[:,0], kps[:,20] - kps[:,17], kps[:,13] - kps[:,17])

        # For adduction, we replace it with the angle between tip & palm since all adduction is from the mcp-joint of the robot hand, thus share the same trend with angle from neighbour joint but much more significant.
        # joint_thumb_adduction = calculate_signed_angle_between_planes(kps[:,4] - kps[:,2], kps[:,9] - kps[:,2], kps[:,13] - kps[:,2], palm_in_norm, direct_define_v2=True)
        
        joint_thumb_adduction = calculate_signed_angle_between_planes(kps[:,4] - kps[:,2], kps[:,5] - kps[:,2], kps[:,13] - kps[:,2], palm_in_norm, direct_define_v2=True)
        joint_index_adduction = calculate_signed_angle_between_planes(kps[:,8] - kps[:,5], kps[:,5] - kps[:,0], kps[:,9] - kps[:,0], palm_in_norm, direct_define_v2=True)
        joint_middle_adduction = calculate_signed_angle_between_planes(kps[:,12] - kps[:,9], kps[:,9] - kps[:,0], kps[:,13] - kps[:,0], palm_in_norm, direct_define_v2=True)
        joint_ring_adduction = calculate_signed_angle_between_planes(kps[:,16] - kps[:,13], kps[:,13] - kps[:,0], kps[:,17] - kps[:,0], palm_in_norm, direct_define_v2=True)
        joint_pinky_adduction = calculate_signed_angle_between_planes(kps[:,20] - kps[:,17], kps[:,17] - kps[:,0], kps[:,13] - kps[:,0], palm_in_norm, direct_define_v2=True)

        # For thumb ip / pip
        joint_thumb_ip = calculate_signed_angle_between_vectors(kps[:,3] - kps[:,2], kps[:,4] - kps[:,3], kps[:,3] - kps[:,5])
        joint_index_pip = calculate_signed_angle_between_vectors(kps[:,6] - kps[:,5], kps[:,7] - kps[:,6], np.cross(kps[:,6] - kps[:,5], kps[:,7] - kps[:,6]))
        joint_middle_pip = calculate_signed_angle_between_vectors(kps[:,10] - kps[:,9], kps[:,11] - kps[:,10], np.cross(kps[:,10] - kps[:,9], kps[:,11] - kps[:,10]))
        joint_ring_pip = calculate_signed_angle_between_vectors(kps[:,14] - kps[:,13], kps[:,15] - kps[:,14], np.cross(kps[:,14] - kps[:,13], kps[:,15] - kps[:,14]))
        joint_pinky_pip = calculate_signed_angle_between_vectors(kps[:,18] - kps[:,17], kps[:,19] - kps[:,18], np.cross(kps[:,18] - kps[:,17], kps[:,19] - kps[:,18]))

        # For dip
        joint_index_dip = calculate_signed_angle_between_vectors(kps[:,7] - kps[:,6], kps[:,8] - kps[:,7], np.cross(kps[:,7] - kps[:,6], kps[:,8] - kps[:,7]))
        joint_middle_dip = calculate_signed_angle_between_vectors(kps[:,11] - kps[:,10], kps[:,12] - kps[:,11], np.cross(kps[:,11] - kps[:,10], kps[:,12] - kps[:,11]))
        joint_ring_dip = calculate_signed_angle_between_vectors(kps[:,15] - kps[:,14], kps[:,16] - kps[:,15], np.cross(kps[:,15] - kps[:,14], kps[:,16] - kps[:,15]))
        joint_pinky_dip = calculate_signed_angle_between_vectors(kps[:,19] - kps[:,18], kps[:,20] - kps[:,19], np.cross(kps[:,19] - kps[:,18], kps[:,20] - kps[:,19]))

        # others
        joint_palm = np.pi / 2 - calculate_signed_angle_between_vectors(kps[:,2] - kps[:,1], kps[:,17] - kps[:,1], np.cross(kps[:,2] - kps[:,1], kps[:,17] - kps[:,1]))
        joint_palm_adduction = calculate_signed_angle_between_planes(kps[:,4] - kps[:,1], kps[:,9] - kps[:,1], kps[:,13] - kps[:,1], palm_in_norm, direct_define_v2=True)

    else:
        # Calculate the left hand joints in wrist coordinate, need to use relative action during pretraining
        palm_in_norm = np.cross(kps[:,13] - kps[:,0], kps[:,5] - kps[:,0])

        joint_thumb_mcp = calculate_signed_angle_between_vectors(kps[:,2] - kps[:,1], kps[:,3] - kps[:,2], kps[:,5] - kps[:,2])
        # For mcp, we replace it with the angle between tip & palm to adopt better to 6DoF robot hand  (the change of angle from neighbour joint is to small)
        joint_index_mcp = calculate_signed_angle_between_vectors(kps[:,5] - kps[:,0], kps[:,8] - kps[:,5], kps[:,9] - kps[:,5])
        joint_middle_mcp = calculate_signed_angle_between_vectors(kps[:,9] - kps[:,0], kps[:,12] - kps[:,9], kps[:,13] - kps[:,9])
        joint_ring_mcp = calculate_signed_angle_between_vectors(kps[:,13] - kps[:,0], kps[:,16] - kps[:,13], kps[:,17] - kps[:,13])
        joint_pinky_mcp = calculate_signed_angle_between_vectors(kps[:,17] - kps[:,0], kps[:,20] - kps[:,17], kps[:,17] - kps[:,13])

        # For adduction, we replace it with the angle between tip & palm since all adduction is from the mcp-joint of the robot hand, thus share the same trend with angle from neighbour joint but much more significant.
        joint_thumb_adduction = calculate_signed_angle_between_planes(kps[:,4] - kps[:,2], kps[:,5] - kps[:,2], kps[:,13] - kps[:,2], palm_in_norm, direct_define_v2=False)
        joint_index_adduction = calculate_signed_angle_between_planes(kps[:,8] - kps[:,5], kps[:,5] - kps[:,0], kps[:,9] - kps[:,0], palm_in_norm, direct_define_v2=False)
        joint_middle_adduction = calculate_signed_angle_between_planes(kps[:,12] - kps[:,9], kps[:,9] - kps[:,0], kps[:,13] - kps[:,0], palm_in_norm, direct_define_v2=False)
        joint_ring_adduction = calculate_signed_angle_between_planes(kps[:,16] - kps[:,13], kps[:,13] - kps[:,0], kps[:,17] - kps[:,0], palm_in_norm, direct_define_v2=False)
        joint_pinky_adduction = calculate_signed_angle_between_planes(kps[:,20] - kps[:,17], kps[:,17] - kps[:,0], kps[:,13] - kps[:,0], palm_in_norm, direct_define_v2=False)

        # For thumb ip / pip
        joint_thumb_ip = calculate_signed_angle_between_vectors(kps[:,3] - kps[:,2], kps[:,4] - kps[:,3], kps[:,5] - kps[:,3])
        joint_index_pip = calculate_signed_angle_between_vectors(kps[:,6] - kps[:,5], kps[:,7] - kps[:,6], np.cross(kps[:,6] - kps[:,5], kps[:,7] - kps[:,6]))
        joint_middle_pip = calculate_signed_angle_between_vectors(kps[:,10] - kps[:,9], kps[:,11] - kps[:,10], np.cross(kps[:,10] - kps[:,9], kps[:,11] - kps[:,10]))
        joint_ring_pip = calculate_signed_angle_between_vectors(kps[:,14] - kps[:,13], kps[:,15] - kps[:,14], np.cross(kps[:,14] - kps[:,13], kps[:,15] - kps[:,14]))
        joint_pinky_pip = calculate_signed_angle_between_vectors(kps[:,18] - kps[:,17], kps[:,19] - kps[:,18], np.cross(kps[:,18] - kps[:,17], kps[:,19] - kps[:,18]))

        # For dip
        joint_index_dip = calculate_signed_angle_between_vectors(kps[:,7] - kps[:,6], kps[:,8] - kps[:,7], np.cross(kps[:,7] - kps[:,6], kps[:,8] - kps[:,7]))
        joint_middle_dip = calculate_signed_angle_between_vectors(kps[:,11] - kps[:,10], kps[:,12] - kps[:,11], np.cross(kps[:,11] - kps[:,10], kps[:,12] - kps[:,11]))
        joint_ring_dip = calculate_signed_angle_between_vectors(kps[:,15] - kps[:,14], kps[:,16] - kps[:,15], np.cross(kps[:,15] - kps[:,14], kps[:,16] - kps[:,15]))
        joint_pinky_dip = calculate_signed_angle_between_vectors(kps[:,19] - kps[:,18], kps[:,20] - kps[:,19], np.cross(kps[:,19] - kps[:,18], kps[:,20] - kps[:,19]))

        # others
        joint_palm = np.pi / 2 - calculate_signed_angle_between_vectors(kps[:,17] - kps[:,1], kps[:,2] - kps[:,1], np.cross(kps[:,17] - kps[:,1], kps[:,2] - kps[:,1]))
        joint_palm_adduction = calculate_signed_angle_between_planes(kps[:,4] - kps[:,1], kps[:,9] - kps[:,1], kps[:,13] - kps[:,1], palm_in_norm, direct_define_v2=False)

    joint_angles = {
        'thumb_mcp': joint_thumb_mcp,
        'thumb_ip': joint_thumb_ip,
        'index_mcp': joint_index_mcp,
        'index_pip': joint_index_pip,
        'index_dip': joint_index_dip,
        'middle_mcp': joint_middle_mcp,
        'middle_pip': joint_middle_pip,
        'middle_dip': joint_middle_dip,
        'ring_mcp': joint_ring_mcp,
        'ring_pip': joint_ring_pip,
        'ring_dip': joint_ring_dip,
        'pinky_mcp': joint_pinky_mcp,
        'pinky_pip': joint_pinky_pip,
        'pinky_dip': joint_pinky_dip,
        'thumb_adduction': joint_thumb_adduction,
        'index_adduction': joint_index_adduction,
        'middle_adduction': joint_middle_adduction,
        'ring_adduction': joint_ring_adduction,
        'pinky_adduction': joint_pinky_adduction,
        'palm': joint_palm,
        'palm_adduction': joint_palm_adduction,
    }

    if mano_joint_concat_order is None:
        mano_joint_concat_order = MANO_JOINT_CONCAT_ORDER

    if return_concat_joints is False:
        joint_results = joint_angles
    else:
        joint_results = []
        for joint_key in mano_joint_concat_order:
            joint_results.append(joint_angles[joint_key])
        joint_results = np.stack(joint_results, axis=-1)
    
    return joint_results, kps

