import os
import argparse
from nuscenes.nuscenes import NuScenes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file to read dataroot from')
    args = parser.parse_args()
    
    # Read dataroot from config (a bit hacky but works without mmengine import)
    data_root = '/home/utkarsh/Desktop/MTP/data/nuscenes'
    with open(args.config, 'r') as f:
        for line in f:
            if 'data_root' in line and '=' in line:
                val = line.split('=')[1].strip().strip("'").strip('"')
                if 'nuscenes' in val:
                    data_root = val
                    break

    print(f"Using dataroot: {data_root}")
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    
    # Read valid scenes from our split to avoid FileNotFoundError
    with open('splits/train_scenes.txt', 'r') as f:
        valid_scenes = set(line.strip() for line in f if line.strip())
        
    valid_scene_objs = [s for s in nusc.scene if s['name'] in valid_scenes]
    
    if len(valid_scene_objs) < 2:
        print("Not enough valid scenes found!")
        return

    # Pick 2 valid scenes
    sample_1 = nusc.get('sample', valid_scene_objs[0]['first_sample_token'])
    sample_2 = nusc.get('sample', valid_scene_objs[1]['first_sample_token'])
    
    samples = {
        'scene1': sample_1,
        'scene2': sample_2
    }
    
    cameras = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    
    cam_to_radar = {
        'CAM_FRONT': 'RADAR_FRONT',
        'CAM_FRONT_LEFT': 'RADAR_FRONT_LEFT',
        'CAM_FRONT_RIGHT': 'RADAR_FRONT_RIGHT',
        'CAM_BACK': 'RADAR_BACK_LEFT',
        'CAM_BACK_LEFT': 'RADAR_BACK_LEFT',
        'CAM_BACK_RIGHT': 'RADAR_BACK_RIGHT'
    }
    
    os.makedirs('projection_checks', exist_ok=True)
    
    for sample_name, sample in samples.items():
        for cam in cameras:
            # Render LiDAR
            lidar_out = f'projection_checks/{sample_name}_lidar_in_{cam.lower()}.png'
            try:
                nusc.render_pointcloud_in_image(
                    sample['token'], 
                    pointsensor_channel='LIDAR_TOP', 
                    camera_channel=cam, 
                    out_path=lidar_out
                )
                print(f"Saved {lidar_out}")
            except Exception as e:
                print(f"Failed to render LiDAR for {cam}: {e}")
            
            # Render Radar
            radar = cam_to_radar[cam]
            radar_out = f'projection_checks/{sample_name}_radar_{radar.lower()}_in_{cam.lower()}.png'
            try:
                nusc.render_pointcloud_in_image(
                    sample['token'], 
                    pointsensor_channel=radar, 
                    camera_channel=cam, 
                    out_path=radar_out
                )
                print(f"Saved {radar_out}")
            except Exception as e:
                print(f"Failed to render Radar {radar} for {cam}: {e}")

if __name__ == '__main__':
    main()
