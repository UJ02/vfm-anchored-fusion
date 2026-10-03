_base_ = ['./centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=200)

lr = 1e-4
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        eta_min=lr * 10,
        begin=0,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=120,
        eta_min=lr * 1e-4,
        begin=80,
        end=200,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=80,
        eta_min=0.85 / 0.95,
        begin=0,
        end=80,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=120,
        eta_min=1,
        begin=80,
        end=200,
        by_epoch=True,
        convert_to_iter_based=True)
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Redefine train pipeline without augmentations
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

data_root = 'data/nuscenes/'
train_dataloader = dict(
    batch_size=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='NuScenesDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_overfit.pkl',
        pipeline=train_pipeline,
        metainfo=dict(version='v1.0-mini', classes=class_names),
        test_mode=False,
        data_prefix=dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR'
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_overfit.pkl',
        pipeline=test_pipeline,
        metainfo=dict(version='v1.0-mini', classes=class_names),
        test_mode=True,
        data_prefix=dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR'
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes_infos_overfit.pkl',
        pipeline=test_pipeline,
        metainfo=dict(version='v1.0-mini', classes=class_names),
        test_mode=True,
        data_prefix=dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
        box_type_3d='LiDAR'
    )
)

val_evaluator = dict(
    type='NuScenesMetric',
    ann_file='data/nuscenes/nuscenes_infos_overfit.pkl',
    jsonfile_prefix='work_dirs/vfm_lidar_overfit/eval'
)
test_evaluator = dict(
    type='NuScenesMetric',
    ann_file='data/nuscenes/nuscenes_infos_overfit.pkl',
    jsonfile_prefix='work_dirs/vfm_lidar_overfit/eval'
)
