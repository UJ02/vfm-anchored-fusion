import os
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def main(dataroot, out_dir):
    print("Loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    
    print("Mapping samples to scenes...")
    sample_to_scene = {}
    for samp in nusc.sample:
        sample_to_scene[samp['token']] = samp['scene_token']
        
    print("Grouping sample_data by scene...")
    scene_to_sd_paths = {scene['token']: [] for scene in nusc.scene}
    for sd in nusc.sample_data:
        samp_token = sd['sample_token']
        scene_token = sample_to_scene[samp_token]
        # get absolute path
        sd_path = nusc.get_sample_data_path(sd['token'])
        scene_to_sd_paths[scene_token].append(sd_path)
        
    print("Checking for missing files on disk...")
    complete_scenes = []
    for scene in nusc.scene:
        scene_token = scene['token']
        paths = scene_to_sd_paths[scene_token]
        missing = False
        for p in paths:
            if not os.path.exists(p):
                missing = True
                break
        if not missing and len(paths) > 0:
            complete_scenes.append(scene['name'])
            
    print(f"Found {len(complete_scenes)} complete scenes out of {len(nusc.scene)}.")
    
    # Split using official splits
    splits = create_splits_scenes()
    train_split = set(splits['train'])
    val_split = set(splits['val'])
    
    train_complete = [s for s in complete_scenes if s in train_split]
    val_complete = [s for s in complete_scenes if s in val_split]
    
    print(f"Train scenes: {len(train_complete)}")
    print(f"Val scenes: {len(val_complete)}")
    
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train_scenes.txt'), 'w') as f:
        for s in train_complete:
            f.write(f"{s}\n")
            
    with open(os.path.join(out_dir, 'val_scenes.txt'), 'w') as f:
        for s in val_complete:
            f.write(f"{s}\n")
            
    print(f"Saved splits to {out_dir}/train_scenes.txt and {out_dir}/val_scenes.txt")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/home/utkarsh/Desktop/MTP/data/nuscenes')
    parser.add_argument('--out_dir', type=str, default='splits')
    args = parser.parse_args()
    main(args.dataroot, args.out_dir)
