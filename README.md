# VFM Anchored Fusion

This repository contains experimental code for VFM Anchored Fusion, a LiDAR-Radar-Camera 3D object detection system built on top of MMDetection3D.

## Installation

This repository depends on MMDetection3D (v1.4.0) being installed in your environment, along with specific modifications documented in `patches/mmdet3d.patch`.

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate mmdet3d
   ```

2. Install pip dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install MMDetection3D and apply patches if compiling from source.

## Repository Structure

- `configs/`: MMDetection3D style configurations for our custom models.
- `projects/vfm_fusion/`: Custom modules (data loaders, tokenizers, fusion modules).
- `tools/`: Training and evaluation scripts.
- `scripts/`: Utility scripts for visualization and debugging.
- `patches/`: Notes and diffs for changes made to MMDetection3D source code.

## Progress

- [x] Phase 1: Environment setup, projection checks, LiDAR overfit test passing.
