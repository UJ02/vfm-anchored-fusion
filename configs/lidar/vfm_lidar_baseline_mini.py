_base_ = ['./centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
train_dataloader = dict(batch_size=2)

# Fix validation dataset metainfo and evaluator version for v1.0-mini
val_dataloader = dict(dataset=dict(metainfo=dict(version='v1.0-mini')))
test_dataloader = dict(dataset=dict(metainfo=dict(version='v1.0-mini')))

val_evaluator = dict(type='NuScenesMetric')
test_evaluator = dict(type='NuScenesMetric')
