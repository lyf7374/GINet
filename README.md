# Geometric-Information-Assisted Network for Robust Skull-Stripping

## Abstract

Skull stripping—the process of isolating brain tissue from non-brain structures in MRI data—is a critical step in neuro-imaging workflows. Although deep-learning methods have shown promise in brain extraction, they often struggle to generalize across diverse datasets because of patient variability, scanner differences, and pathological conditions. This limitation stems from their reliance on local texture patterns, which makes them sensitive to input variations.

We introduce a **Geometric-Information-Assisted Network (GINet)** that improves skull stripping by fusing geometric information—a texture-invariant prior—into the segmentation pipeline, mitigating variability across subjects and datasets. GINet comprises three primary components:

1. **GI filtering module** – efficiently extracts lightweight geometric priors from input scans.  
2. **GI adaptation module** – adapts a standard brain template to input-specific GI while preserving structural integrity.  
3. **Deep GI-fused segmentation module** – hierarchically integrates GI with image features for robust segmentation.

To further enhance performance, we propose a *multi-depth (MD)* loss that refines feature maps with adaptive GI at multiple scales. Trained exclusively on healthy data, GINet generalizes effectively to unseen healthy and pathological datasets, outperforming classical and state-of-the-art skull-stripping methods.

## Project Structure

The directory structure is organized as follows:

- `datasets/` — GI support generation utilities  
  - `GI_generate_support.py` — Script to create geometric priors
- `model/` — Model architecture definitions  
  - Contains core GINet architecture components
- `utils/` — Utility functions and helper scripts
- `GINet_train.py` — Training script for GINet
- `GI_extration.ipynb` — Jupyter notebook for GI extraction
- `evaluation.py` — Evaluation script on multiple datasets
- `template/` —Standard MNI152 brain template
- `README.md` — This documentation file

## Usage

### Download Datasets
#### for training
CC359: https://www.ccdataset.com/download
#### for evaluation
NFBS: http://preprocessed-connectomes-project.org/NFB_skullstripped/
LPBA: https://loni.usc.edu/research/atlases
### 1. Extract Geometric Information (GI)
GI_extration.ipynb
with datasets/GI_generate_support.py

### 2. Train GINet
python GINet_train.py

### 3. Evaluate the Model
python evaluation.py

## Status

This repository is currently under active development. We will complete and release the full version soon. Stay tuned!



