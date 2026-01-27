# iSAID Instance Segmentation

Custom Mask R-CNN implementation for instance segmentation on aerial/satellite imagery using the iSAID dataset.

## Features

- **Custom Architecture**: Mask R-CNN with EfficientNet-B0 backbone and CBAM attention modules
- **Optimized for Aerial Images**: Custom anchor sizes for small object detection
- **Experiment Tracking**: Full Weights & Biases integration
- **Flexible Training**: CLI-based training script with configuration file support

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/michaelo-ponteski/isaid-instance-segmentation.git
cd isaid-instance-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the iSAID dataset patches and organize as:
```
iSAID_patches/
├── train/
│   ├── images/
│   └── instances_only_filtered_train.json
└── val/
    ├── images/
    └── instances_only_filtered_val.json
```

### 3. Training

```bash
# Basic training
python train.py --config config.yaml --data-root /path/to/iSAID_patches

# Training with custom parameters
python train.py \
    --config config.yaml \
    --data-root /path/to/iSAID_patches \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0003

# Training without W&B logging
python train.py --config config.yaml --data-root /path/to/data --no-wandb

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/last.pth
```

### 4. Configuration

Edit `config.yaml` to customize training:

```yaml
data:
  root_dir: "iSAID_patches"
  num_classes: 16
  image_size: 800

training:
  epochs: 25
  batch_size: 8
  learning_rate: 0.0003

wandb:
  enabled: true
  project: "isaid-custom-segmentation"
```

## Project Structure

```
├── config.yaml           # Configuration file
├── train.py              # Training script
├── requirements.txt      # Dependencies
├── REPORT.md             # Project report
├── datasets/             # Dataset loader
├── models/               # Model architecture
├── training/             # Training utilities
└── notebooks/            # Jupyter notebooks
```

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{isaid2019,
  title={iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images},
  author={Zamir, Waqas and others},
  booktitle={CVPR Workshops},
  year={2019}
}
```
