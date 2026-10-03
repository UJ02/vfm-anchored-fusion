import pickle
import numpy as np
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_info', default='/home/utkarsh/Desktop/MTP/data/nuscenes/nuscenes_infos_train.pkl')
    parser.add_argument('--val_info', default='/home/utkarsh/Desktop/MTP/data/nuscenes/nuscenes_infos_val.pkl')
    args = parser.parse_args()

    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]

    for split_name, file_path in [('Train', args.train_info), ('Val', args.val_info)]:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            infos = data['data_list']
            print(f"=== {split_name} Split ({len(infos)} samples) ===")
            
            counter = Counter()
            for info in infos:
                gt_names = info.get('gt_names', [])
                valid_flag = info.get('valid_flag', [])
                
                # We only count valid instances
                for name, valid in zip(gt_names, valid_flag):
                    if valid:
                        counter[name] += 1
            
            for cls in class_names:
                print(f"{cls}: {counter.get(cls, 0)}")
            print()
            
        except FileNotFoundError:
            print(f"File {file_path} not found. Ensure create_data.py finished successfully.")

if __name__ == '__main__':
    main()
