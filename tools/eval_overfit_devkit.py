import json
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
import nuscenes.eval.detection.evaluate as nusc_eval_module
import nuscenes.eval.common.loaders as nusc_loaders

# Load original load_gt
original_load_gt = nusc_loaders.load_gt

def patched_load_gt(*args, **kwargs):
    gt_boxes = original_load_gt(*args, **kwargs)
    
    # Load our prediction tokens
    with open('work_dirs/vfm_lidar_overfit/eval/pred_instances_3d/results_nusc.json', 'r') as f:
        preds = json.load(f)
    pred_tokens = set(preds['results'].keys())
    
    # Filter gt_boxes
    filtered_tokens = []
    for token in gt_boxes.sample_tokens:
        if token in pred_tokens:
            filtered_tokens.append(token)
        else:
            del gt_boxes.boxes[token]
    
    # Now gt_boxes only contains the 8 tokens
    return gt_boxes

# Apply patch
nusc_eval_module.load_gt = patched_load_gt
nusc_loaders.load_gt = patched_load_gt

def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
    cfg = config_factory('detection_cvpr_2019')
    result_path = 'work_dirs/vfm_lidar_overfit/eval/pred_instances_3d/results_nusc.json'
    
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=result_path,
        eval_set='mini_train',
        output_dir='work_dirs/vfm_lidar_overfit/eval',
        verbose=True
    )
    
    metrics = nusc_eval.main(render_curves=False)
    print("NDS:", metrics[0].nd_score)
    print("mAP:", metrics[0].mean_ap)

if __name__ == '__main__':
    main()
