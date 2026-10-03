import argparse
import os
from os import path as osp
import numpy as np
import mmengine
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from mmdet3d.datasets.convert_utils import NuScenesNameMapping

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path: 
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep

def create_custom_nuscenes_infos(root_path,
                                 info_prefix,
                                 train_scenes,
                                 val_scenes,
                                 version='v1.0-trainval',
                                 max_sweeps=10):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes)))
    
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos_with_radar(
        nusc, train_scenes, val_scenes, test=False, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    print('train sample: {}, val sample: {}'.format(
        len(train_nusc_infos), len(val_nusc_infos)))
    
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(root_path, '{}_infos_train.pkl'.format(info_prefix))
    mmengine.dump(data, info_path)
    
    data['infos'] = val_nusc_infos
    info_val_path = osp.join(root_path, '{}_infos_val.pkl'.format(info_prefix))
    mmengine.dump(data, info_val_path)


def _fill_trainval_infos_with_radar(nusc,
                                    train_scenes,
                                    val_scenes,
                                    test=False,
                                    max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []

    radar_sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

    for sample in mmengine.track_iter_progress(nusc.sample):
        if sample['scene_token'] not in train_scenes and sample['scene_token'] not in val_scenes:
            continue

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmengine.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'radars': dict(),
            'radar_sweeps': {radar: [] for radar in radar_sensors},
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # LiDAR sweeps
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # Radar sweeps and current frame
        for radar in radar_sensors:
            radar_token = sample['data'][radar]
            radar_path, _, radar_intrinsic = nusc.get_sample_data(radar_token)
            radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, radar)
            radar_info.update(cam_intrinsic=radar_intrinsic)
            info['radars'].update({radar: radar_info})

            radar_sd_rec = nusc.get('sample_data', radar_token)
            radar_sweeps = []
            while len(radar_sweeps) < max_sweeps:
                if not radar_sd_rec['prev'] == '':
                    sweep = obtain_sensor2top(nusc, radar_sd_rec['prev'], l2e_t,
                                              l2e_r_mat, e2g_t, e2g_r_mat, radar)
                    radar_sweeps.append(sweep)
                    radar_sd_rec = nusc.get('sample_data', radar_sd_rec['prev'])
                else:
                    break
            info['radar_sweeps'][radar] = radar_sweeps

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations],
                dtype=bool).reshape(-1)
            
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesNameMapping:
                    names[i] = NuScenesNameMapping[names[i]]
            names = np.array(names)
            
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

            if 'lidarseg' in nusc.table_names:
                info['pts_semantic_mask_path'] = osp.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def load_split(split_file):
    if not osp.exists(split_file):
        raise FileNotFoundError(f"{split_file} not found")
    with open(split_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/home/utkarsh/Desktop/MTP/data/nuscenes')
    parser.add_argument('--out-dir', type=str, default='/home/utkarsh/Desktop/MTP/data/nuscenes')
    parser.add_argument('--train-split-file', type=str, default='splits/train_scenes.txt')
    parser.add_argument('--val-split-file', type=str, default='splits/val_scenes.txt')
    parser.add_argument('--extra-tag', type=str, default='nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--max-sweeps', type=int, default=10)
    args = parser.parse_args()

    train_scenes = load_split(args.train_split_file)
    val_scenes = load_split(args.val_split_file)

    create_custom_nuscenes_infos(
        args.dataroot,
        args.extra_tag,
        train_scenes,
        val_scenes,
        version=args.version,
        max_sweeps=args.max_sweeps
    )
    print("Infos generated successfully.")

if __name__ == '__main__':
    main()
