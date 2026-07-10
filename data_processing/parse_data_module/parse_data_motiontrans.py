# ===============================================================
# Robot Teleoperation Data Alignment
# support: image, stereo, pointclouds (from sensor)
# ===============================================================

import pickle
import numpy as np
import cv2
import pyzed.sl as sl
from typing import Sequence, Tuple, Dict, Optional, Union, Generator
import os
import pathlib
import random
import yaml
import click
import shutil
from copy import deepcopy
from multiprocessing import Process
from tqdm import tqdm
from common.cv2_utils import get_image_transform_resize_crop
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.replay_buffer import ReplayBuffer
from common.svo_utils import SVOReader
from scipy.spatial.transform import Rotation as R
from common.interpolation_utils import PoseInterpolator, get_interp1d
from common.pose_utils import pose_to_mat, mat_to_pose
from common.gpt_instruction_expansion import get_openai_client, get_openai_instruction_expansion_episodes



def egocentric_to_base_obs_transformation(pose2cam, cam2base, inv_cam2base=False):
    """Transform the observation from the egocentric view to the base view.

    Args:
        pose: The pose of the camera in the base frame.  (T, 6), rotvec
        cam2base: The pose of the camera in the base frame.  (4, 4)
        inv_cam2base: If True, cam2base need to be inverted before using.

    Returns:
        The transformed observation.
    """
    episode_length = len(pose2cam)
    cam_to_base_mat = np.array(cam2base)

    gripper_to_cam = pose2cam
    gripper_to_cam_mat = np.zeros((episode_length, 4, 4))
    gripper_to_cam_mat[:, :3, :3] = R.from_rotvec(gripper_to_cam[:, 3:]).as_matrix()
    gripper_to_cam_mat[:, :3, 3] = gripper_to_cam[:, :3]
    gripper_to_cam_mat[:, -1, -1] = 1.0
         
    if not inv_cam2base:
        gripper_to_base_mat = cam_to_base_mat[None] @ gripper_to_cam_mat
    else:
        gripper_to_base_mat = np.linalg.inv(cam_to_base_mat[None]) @ gripper_to_cam_mat
    gripper_to_base = np.zeros((episode_length, 6))
    gripper_to_base[:, :3] = gripper_to_base_mat[:, :3, 3]
    gripper_to_base[:, 3:] = R.from_matrix(gripper_to_base_mat[:, :3, :3]).as_rotvec()

    return gripper_to_base


def get_eef_pos_velocity(eef_pos_seq):
    delta = np.linalg.norm(eef_pos_seq[1:] - eef_pos_seq[:-1], axis=-1)
    vel = delta.mean()
    return vel


def conversion_single_trajectory(
        hand_to_eef,
        mode,
        save_dir,
        out_resolutions_resize: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        out_resolutions_crop: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        out_resolutions_image_final: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
):
    cfg_path = os.path.join(save_dir, "episode_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    n_robot = len(cfg['robots'])

    episode_path = os.path.join(save_dir, "episode.pkl")
    with open(episode_path, "rb") as f:
        episode = pickle.load(f)

    # remove the timestamp where robot0_eef_pos does not change
    start_time_idx = 0
    eps = 0.005
    for i in range(1, len(episode[f'robot0_eef_pos'])):
        if np.linalg.norm(episode[f'robot0_eef_pos'][i] - episode[f'robot0_eef_pos'][0]) > eps:
            start_time_idx = i
            break
    for key in episode.keys():
        episode[key] = episode[key][start_time_idx:]

    dt = (episode['timestamp'][1:] - episode['timestamp'][:-1]).mean()

    # ========================== Remove Initialization Stage Data =================================

    t_init = 0
    while (episode['action'][t_init] == 0.).all():
        t_init += 1
    for key in episode.keys():
        episode[key] = episode[key][t_init:]
    episode_length = len(episode['timestamp'])

    # ========================== Transformation to Egocentric View =================================
    assert len(cfg['cameras']) == 1  # TODO: support multiple cameras
    n_robot = len(cfg['robots'])
    st_action_dim_idx = 0
    for robot_id in range(n_robot):
        cam_to_base = cfg['cameras'][0]['calib_cam_to_base'][robot_id]
        base_id, cam2base_pose = cam_to_base[0], cam_to_base[1]

        assert base_id == robot_id
        action_dim = cfg['robots'][robot_id]['action_dim']

        assert action_dim == 6
        gripper_to_base = episode['action'][:, st_action_dim_idx: st_action_dim_idx+action_dim]
        gripper_to_base = mat_to_pose(pose_to_mat(gripper_to_base) @ hand_to_eef)
        gripper_to_cam = egocentric_to_base_obs_transformation(pose2cam=gripper_to_base, cam2base=cam2base_pose,
                                                               inv_cam2base=True)
        episode['action'][:, st_action_dim_idx: st_action_dim_idx+action_dim] = gripper_to_cam

        gripper_to_base = np.concatenate([episode[f'robot{robot_id}_eef_pos'], episode[f'robot{robot_id}_eef_rot_axis_angle']], axis=1)
        gripper_to_base = mat_to_pose(pose_to_mat(gripper_to_base) @ hand_to_eef)
        gripper_to_cam = egocentric_to_base_obs_transformation(pose2cam=gripper_to_base, cam2base=cam2base_pose,
                                                               inv_cam2base=True)
        episode[f'robot{robot_id}_eef_pos'] = gripper_to_cam[:, :3]
        episode[f'robot{robot_id}_eef_rot_axis_angle'] = gripper_to_cam[:, 3:]

        st_action_dim_idx = st_action_dim_idx + action_dim + cfg['grippers'][robot_id]['action_dim']

    # ========================== Action Class Initialization =================================
    ### robot -> gripper, right_hand -> left_hand
    st_action_dim_idx = 0
    actions = []
    for robot_id in range(n_robot):
        action_dim = cfg['robots'][robot_id]['action_dim']
        assert action_dim == 6
        actions.append(episode[f'robot{robot_id}_eef_pos'])
        actions.append(episode[f'robot{robot_id}_eef_rot_axis_angle'])
        st_action_dim_idx += action_dim

        action_dim = cfg['grippers'][robot_id]['action_dim']
        actions.append(episode[f'gripper{robot_id}_gripper_pose'])
        st_action_dim_idx += action_dim

    for i in range(len(actions)):
        if actions[i].ndim == 3:
            actions[i] = actions[i][..., 0]
    actions = np.concatenate(actions, axis=-1)
    episode['action'] = actions

    # ========================== Camera Action Alignment =================================
    svo_path = os.path.join(save_dir, "videos", "recording.svo2")
    svo_stereo, svo_pointcloud = False, False
    if mode in ['p', 'a']:
        svo_pointcloud = True
    if mode in ['s', 'a']:
        svo_stereo = True

    serial_id = str(cfg['cameras'][0]['device_id'])
    svo_camera = SVOReader(svo_path, serial_number=serial_id)
    svo_camera.set_reading_parameters(image=True, depth=False, pointcloud=svo_pointcloud, concatenate_images=False)
    frame_count = svo_camera.get_frame_count()
    width, height = svo_camera.get_frame_resolution()

    next_global_idx = 0
    start_time = episode['timestamp'][0]

    obs_dict = dict()
    episode['camera_ego_real_timestamp'] = np.zeros((episode_length,), dtype=np.float64)
    transform_img = get_image_transform_resize_crop(input_res=(width, height), output_resize_res=out_resolutions_resize,
                                                    output_crop_res=out_resolutions_crop, bgr_to_rgb=True)
    obs_dict['rgb'] = ('image', f'{serial_id}_left', transform_img)
    if svo_stereo:
        obs_dict['rgb_right'] = ('image', f'{serial_id}_right', transform_img)
    if svo_pointcloud:
        transform_pointcloud = get_image_transform_resize_crop(input_res=(width, height),
                                                               output_resize_res=out_resolutions_resize,
                                                               output_crop_res=out_resolutions_crop, is_pcd=True)
        obs_dict['pointcloud'] = ('pointcloud', f'{serial_id}_left', transform_pointcloud)

    frame_cut_fp = os.path.join(save_dir, "frame_cut.txt")
    frame_cut = None
    if os.path.exists(frame_cut_fp):
        # read the number in txt
        with open(frame_cut_fp, "r") as f:
            frame_cut = f.read().strip()
        if frame_cut.isdigit():
            frame_cut = int(frame_cut) - start_time_idx
        else:
            print(f"Frame cut {frame_cut} is not a digit, set to None.")
            frame_cut = None 
    if frame_cut is not None:
        episode_length = min(episode_length, frame_cut)
    for episode_key in episode.keys():
        episode[episode_key] = episode[episode_key][:episode_length]

    global_idx = 0

    # for t in tqdm(range(frame_count)):
    for t in range(frame_count):
        svo_output = svo_camera.read_camera(return_timestamp=True)
        if svo_output is None:
            break
        else:
            data_dict, timestamp = svo_output
            timestamp = timestamp / 1000.0
        if timestamp < episode['timestamp'][0] - dt:
            continue

        local_idxs, global_idxs, next_global_idx \
            = get_accumulate_timestamp_idxs(
            timestamps=[timestamp],
            start_time=start_time,
            dt=dt,
            next_global_idx=next_global_idx
        )

        if len(global_idxs) > 0:
            for global_idx in global_idxs:
                if global_idx >= episode_length:
                    # print(f"Warning: global_idx {global_idx} >= episode_length {episode_length}, break.")
                    break
                for key in obs_dict.keys():
                    value = data_dict[obs_dict[key][0]][obs_dict[key][1]]
                    transform = obs_dict[key][2]
                    if value.shape[-1] == 4:
                        value = value[..., :3]
                    value = transform(value)
                    if 'rgb' in key:
                        value = cv2.resize(value, out_resolutions_image_final, interpolation=cv2.INTER_LINEAR)
                    if key == 'rgb':
                        if 'camera_ego_rgb' not in episode.keys():
                            episode['camera_ego_rgb'] = np.zeros((episode_length,) + value.shape, dtype=value.dtype)
                        episode['camera_ego_rgb'][global_idx] = value
                    elif key == 'pointcloud':
                        if 'camera_ego_pointcloud' not in episode.keys():
                            episode['camera_ego_pointcloud'] = np.zeros((episode_length,) + value.shape, dtype=value.dtype)
                        episode['camera_ego_pointcloud'][global_idx] = value
                    else:
                        # print(f"skip {key} for saving")
                        pass

                episode['camera_ego_real_timestamp'][global_idx] = timestamp
        if next_global_idx == episode_length:
            break

    if (next_global_idx < episode_length) and (global_idx != episode_length):
        abandoned_frames = episode_length - next_global_idx
        for key in episode.keys():
            try:
                episode[key] = episode[key][:-abandoned_frames]
            except:
                pass
        # print(f"Warning: {next_global_idx} < {episode_length}, abandoned {abandoned_frames} frames.")

    n_length = np.min([episode['timestamp'].shape[-1], episode['camera_ego_real_timestamp'].shape[-1]])
    for key in episode.keys():
        episode[key] = episode[key][:n_length]

    # for key in episode.keys():
    #     try:
    #         print(f"Key: {key}, shape: {episode[key].shape}")
    #     except:
    #         print(f"Key: {key}, {episode[key]}")
    
    episode_save = dict()
    T = episode['timestamp'].shape[0]
    episode_save['timestamps'] = episode['timestamp']
    episode_save['camera_ego_rgb'] = episode['camera_ego_rgb']
    episode_save['camera_ego_real_timestamp'] = episode['camera_ego_real_timestamp']
    if 'camera_ego_pointcloud' in episode.keys():
        episode_save['camera_ego_pointcloud'] = episode['camera_ego_pointcloud']
    episode_save['right_wrist_pose'] = np.concatenate([episode['robot0_eef_pos'], episode['robot0_eef_rot_axis_angle']], axis=1)
    episode_save['right_hand_joints'] = episode['gripper0_gripper_pose']
    # inspire_hand 6dof servo-pos  (pinky, ring, middle, index, thumb-curve, thumb-inside)
    episode_save['right_hand_joints_idx'] = np.stack([[22, 17, 12, 7, 2, 1]] * T)
    episode_save['right_tactile_data_fingertorque'] = episode['gripper0_gripper_force'][:, :4].reshape(T, 4, 1)
    episode_save['right_tactile_area_fingertorque'] = np.stack([[20, 19, 18, 17]] * T)
    episode_save['right_tactile_sensor_fingertorque'] = np.array(['InspireHand'] * T)
    episode_save['right_tactile_type_fingertorque'] = np.array(['state'] * T)
    episode_save['right_tactile_data_thumbtorque'] = episode['gripper0_gripper_force'][:, -2:].reshape(T, 1, 2)
    episode_save['right_tactile_area_thumbtorque'] = np.stack([[16]] * T)
    episode_save['right_tactile_sensor_thumbtorque'] = np.array(['InspireHand'] * T)
    episode_save['right_tactile_type_thumbtorque'] = np.array(['state'] * T)

    return episode_save


def conversion_trajectory(input_data_fp_list, hand_to_eef, instruction, client, mode, out_resolutions_resize,
                          out_resolutions_crop, 
                          resolution_image_final,
                          replay_buffer,
                          process_id, verbose,
                          fill_instruction_length=100):
    pbar = tqdm(input_data_fp_list, desc=f"Process {process_id}")

    instruction = instruction.replace('_', ' ')
    if not instruction.endswith('.'):
        instruction = instruction + '.'
    instruction_candidate_list = None
    for input_data_fp in pbar:
        save_dir, source, source_idx = input_data_fp
        episode = conversion_single_trajectory(
            hand_to_eef,
            mode,
            save_dir,
            out_resolutions_resize,
            out_resolutions_crop,
            resolution_image_final
        )
        if instruction_candidate_list is None:
            T = episode['timestamps'].shape[0]
            episode['sub_task_instruction'] = np.array([instruction] * T)

            print(f"Expanding instruction for {save_dir}")
            _, instruction_candidate_list = get_openai_instruction_expansion_episodes(
                episode, 
                image_key='camera_ego_rgb',
                n_image=2,
                instruction_key='sub_task_instruction',
                client=client,
                embodiment='robot',
                n_new_instructions=40,
                fill_instruction_length=100)
            instruction_candidate_list = instruction_candidate_list + [instruction.ljust(fill_instruction_length)] * 10
            print(f"Finished expanding instruction for {save_dir}")

        T = episode['timestamps'].shape[0]
        while True:
            instruction_choice = random.choice(instruction_candidate_list)
            if len(instruction_choice) == fill_instruction_length:
                break
        episode['sub_task_instruction'] = np.array([instruction_choice] * T)
        replay_buffer.add_episode(episode, compressors='disk')  # with lock mechanism inside replay_buffer instance
        pbar.set_postfix(steps=T)


@click.command()
@click.option('--input_dir', '-i', required=True)
@click.option('--output', '-o', required=True)
@click.option('--hand_to_eef_file', '-ehf', required=True)
@click.option('--mode', '-m', required=True, type=click.Choice(['o', 'p', 's', 'a'], case_sensitive=False), default='o',
              help="o: only image, p: with pointcloud, s: with stereo-image, a: with all, including pointcloud and stereo")
@click.option('--resolution_resize', '-ror', default='1280x720')
@click.option('--resolution_crop', '-or', default='640x480')
@click.option('--resolution_image_final', '-for', default='224x224')
@click.option('--num_use_source', '-nus', default=None, type=int)
@click.option('--n_encoding_threads', '-ne', default=-1, type=int)
@click.option('--verbose', '-v', is_flag=True, default=True)
def main(input_dir, output, hand_to_eef_file, 
         mode,
         resolution_resize, resolution_crop, resolution_image_final, num_use_source,
         n_encoding_threads,
         verbose):
    hand_to_eef = np.load(hand_to_eef_file)
    out_resolution_resize = tuple(int(x) for x in resolution_resize.split('x'))
    out_resolution_crop = tuple(int(x) for x in resolution_crop.split('x'))
    resolution_image_final = tuple(int(x) for x in resolution_image_final.split('x'))
    input_dir_list = os.listdir(input_dir)
    input_dir_list = [os.path.join(input_dir, x) for x in input_dir_list]

    client = get_openai_client()

    for idx, input_dir in enumerate(input_dir_list):
        print(f"Processing [{idx+1}/{len(input_dir_list)}]: {input_dir}")
        input_folder = input_dir.split('/')[-1]
        embodiment = input_folder.split('_')[0]
        assert embodiment == "robot"
        environment_setting = input_folder.split('_')[1]
        instruction = '_'.join(input_folder.split('_')[2:])

        replay_buffer_fp = os.path.join(output, embodiment + "_" + environment_setting + "+" + instruction + "+")
        
        if environment_setting == "me":  # multi-environment
            if num_use_source is not None and num_use_source < 0:
                num_use_source = None
            if num_use_source is not None:
                replay_buffer_fp = replay_buffer_fp + "_src" + str(num_use_source)
        replay_buffer_fp = replay_buffer_fp + ".zarr"
        replay_buffer_fp = pathlib.Path(os.path.expanduser(replay_buffer_fp))
        if os.path.exists(replay_buffer_fp):
            shutil.rmtree(replay_buffer_fp)

        replay_buffer = ReplayBuffer.create_from_path(replay_buffer_fp, mode='a')
        if environment_setting == "me":          # multi-environment
            input_data_fp_list = []
            source_list = os.listdir(input_dir)
            source_list.sort()
            for sidx, source in enumerate(source_list):
                if num_use_source is not None and sidx == num_use_source:
                    print(f"Use Source: {source_list[:num_use_source]}")
                    break
                tmp_fp_list = os.listdir(os.path.join(input_dir, source))
                tmp_fp_list = [(os.path.join(input_dir, source, fp), source, sidx) for fp in tmp_fp_list]
                input_data_fp_list = input_data_fp_list + tmp_fp_list
                if sidx + 1 == num_use_source:
                    print(f"Use Source: {source_list[:num_use_source]}")
                    print("Use All Sources.")
                    break
        else:
            input_data_fp_list = os.listdir(input_dir)
            input_data_fp_list.sort()
            input_data_fp_list = [(os.path.join(input_dir, fp), "default", 0) for fp in input_data_fp_list] 

        random.shuffle(input_data_fp_list)

        if n_encoding_threads > 1:
            input_data_fp_batch_list = []
            for i in range(n_encoding_threads):
                input_data_fp_batch_list.append(input_data_fp_list[i::n_encoding_threads])

            process_list = []
            for i in range(n_encoding_threads):
                p = Process(target=conversion_trajectory, args=(
                input_data_fp_batch_list[i], hand_to_eef, instruction, client, mode, out_resolution_resize, out_resolution_crop, resolution_image_final,
                replay_buffer, i, verbose))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            conversion_trajectory(input_data_fp_list, hand_to_eef, instruction, client, mode, out_resolution_resize, out_resolution_crop, resolution_image_final,
                                  replay_buffer, 0, verbose)

        print(f"Saving to disk finish: Task {input_dir}")


if __name__ == "__main__":
    # test_conversion()
    main()

# bash scripts_data/robot_data_conversion_base.sh