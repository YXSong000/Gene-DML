# Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images

This GitHub repository is the official code of paper Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images (accepted by WACV2026). Please kindly refer to our paper: https://arxiv.org/abs/2507.14670.

## Table of Contents

- [Overview](#overview)
- [System](#system)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Feature Extraction](#feature-extraction)
- [Training](#training)
- [Testing](#testing)
- [Configuration](#configuration)
- [Citation](#citation)

## Overview

![Main Figure](overview.png)

## System

- **Python**: 3.11.13 (or compatible)
- **PyTorch**: 2.5.1
- **torchvision**: 0.20.1
- **CUDA**: 12.4 

## Installation

We recommend using the provided conda environment file:

```bash
conda env create -f environment.yml
conda activate gene-dml
```

Install PyTorch with CUDA support (adjust CUDA version based on your system):

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

## Data Preparation

### Step 1: Download Preprocessed Data

Download the preprocessed data from the [st data](https://unisydneyedu-my.sharepoint.com/:u:/g/personal/yson2999_uni_sydney_edu_au/EcxJY-e8hQtFpgv64W0Xn2EBhZrJ7PLbCjjmHeKc-0xrLw?e=evVuKD) link and save it to your data directory:

```bash
# Create data directory
mkdir -p path/to/data

# Download and extract TRIPLEX.zip to path/to/data
# The extracted structure should be:
# data/
#   ├── her2st/
#   ├── skinst/
#   ├── stnet/
#   └── test/
```

### Step 2: Organize Data Structure

Your data directory should follow this structure:

```
data/
├── her2st/                 # HER2ST dataset
│   ├── ST-cnts/            # Gene expression count files
│   ├── ST-imgs/            # Histology images
│   ├── ST-spotfiles/       # Spot coordinate files
│   ├── gt_features_224/    # Target features (extracted by UNI)
│   └── n_features_5_224/   # Neighbor features (extracted by UNI)
├── skinst/                 # skinST dataset
│   └── ...
├── stnet/                  # STNet dataset
│   └── ...
└── test/                   # External test datasets
    └── ...
```

## Feature Extraction

The framework supports multiple feature extractors. In this paper, we utilize UNI model, and the feature extracted by the UNI is included in the downloaded data. Before training, you need to download the UNI model from Hugging Face (https://huggingface.co/MahmoodLab/UNI), or any other feature extractors, saving to weights path ``./weights`` and extract features by following code:

### Using UNI Model

Extract features using the UNI model:

```bash
# For internal cross-validation (HER2ST dataset)
python preprocess/extract_features_uni.py --config her2st/dml --mode internal --extract_mode target 

python preprocess/extract_features_uni.py --config her2st/dml --mode internal --extract_mode neighbor 

# For external test datasets (10x_breast_ff1 dataset)
python preprocess/extract_features_uni.py --mode external --extract_mode target --test_name 10x_breast_ff1 
```

**Parameters:**
- `--config`: Configuration file path (e.g., `her2st/dml`)
- `--mode`: `internal` for cross-validation or `external` for test datasets
- `--extract_mode`: `target` for target patches or `neighbor` for neighbor patches
- `--num_n`: Number of neighbors (default: 5)
- `--test_name`: Test dataset name (for external mode)
- `--gpu`: GPU ID to use


## Training

### Cross-Validation Training

Train models using k-fold cross-validation:

```bash
# HER2ST dataset
python main.py --config_name her2st/dml --mode train --k 18 --use_kmeans --model_size base

# SkinST dataset
python main.py --config_name skinst/dml --mode train --k 25 --use_kmeans --model_size base

# STNet dataset
python main.py --config_name stnet/dml --mode train --k 90 --use_kmeans --model_size base
```

**Key Parameters:**
- `--config_name`: Configuration file path
- `--mode`: `train` for training
- `--gpu`: GPU ID
- `--k`: Number of clusters/pathways
- `--use_kmeans`: Use k-means clustering (recommended)
- `--model_size`: `base` or `large` model architecture 
- `--lambda_cld`: Weight for contrastive learning loss
- `--cld_t`: Temperature for contrastive learning

## Testing

### Internal Test (Cross-Validation)

Test on held-out folds from cross-validation:

```bash
python main.py --config_name her2st/dml --mode test --fold 0 --model_path /path/to/model/ckpt
```

### External Test

Test on independent external datasets:

```bash
python main.py --mode external_test --test_name 10x_breast_ff1 --model_path /path/to/model/ckpt
```

**Available test datasets:**
- `10x_breast_ff1` (breast data)
- `10x_breast_ff2` (breast data)
- `10x_breast_ff3` (breast data)
- `NCBI463` (skin data)
- `NCBI464` (skin data)

## Configuration

Configuration files are located in the `config/` directory. Each dataset has its own configuration files (please edit the config files content accordingly):

```
config/
├── her2st/
│   └── dml.yaml
├── skinst/
│   └── dml.yaml
└── stnet/
    └── dml.yaml
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{song2025genedmldualpathwaymultileveldiscrimination,
      title={Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images}, 
      author={Yaxuan Song and Jianan Fan and Hang Chang and Weidong Cai},
      year={2025},
      eprint={2507.14670},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2507.14670}, 
}
```

## Acknowledgements

- Code for data processing is based on [TRIPLEX](https://github.com/NEXGEM/TRIPLEX).
- Code for feature clustering is based on [CLD](https://github.com/frank-xwang/CLD-UnsupervisedLearning).