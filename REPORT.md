# iSAID Instance Segmentation with Custom Mask R-CNN

## Project Report

**Course:** Computer Vision
**Semester:** 5
**Authors:** [Your Names Here]
**Date:** January 2026
**GitHub Repository:** [https://github.com/YOUR_USERNAME/isaid-instance-segmentation](https://github.com/YOUR_USERNAME/isaid-instance-segmentation)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Problem Description](#3-problem-description)
4. [Model Architecture](#4-model-architecture)
5. [Model Analysis](#5-model-analysis)
6. [Training Description](#6-training-description)
7. [Metrics and Evaluation](#7-metrics-and-evaluation)
8. [Hyperparameters](#8-hyperparameters)
9. [Results and Plots](#9-results-and-plots)
10. [Model Comparison](#10-model-comparison)
11. [Challenges and Solutions](#11-challenges-and-solutions)
12. [Runtime Environment](#12-runtime-environment)
13. [Training and Inference Time](#13-training-and-inference-time)
14. [Libraries and Tools](#14-libraries-and-tools)
15. [Bibliography](#15-bibliography)
16. [Evaluation Checklist](#16-evaluation-checklist)

---

## 1. Introduction

This project implements instance segmentation on aerial/satellite imagery using a custom Mask R-CNN architecture. The goal is to detect and segment objects in the iSAID (Instance Segmentation in Aerial Images Dataset), which contains high-resolution satellite images with dense object annotations.

Instance segmentation combines object detection (bounding boxes + class labels) with semantic segmentation (pixel-level masks), making it one of the most challenging computer vision tasks.

### 1.1 Project Goals

- Implement a custom Mask R-CNN with >50% custom layers
- Integrate attention mechanisms (CBAM) for improved feature extraction
- Optimize anchor configurations for aerial imagery characteristics
- Achieve competitive mAP scores on the iSAID validation set
- Provide comprehensive experiment tracking with Weights & Biases

---

## 2. Dataset Description

### 2.1 iSAID Dataset Overview

The **iSAID (Instance Segmentation in Aerial Images Dataset)** is a large-scale benchmark dataset for instance segmentation in aerial imagery. It is derived from the DOTA dataset and provides pixel-level annotations for instance segmentation.

| Property              | Value                           |
| --------------------- | ------------------------------- |
| **Total Images**      | 2,806 high-resolution images    |
| **Total Instances**   | 655,451 annotated instances     |
| **Image Resolution**  | Varies (up to 4000×4000 pixels) |
| **Patch Size Used**   | 800×800 pixels                  |
| **Number of Classes** | 15 object classes + background  |
| **Annotation Format** | COCO-style JSON                 |

### 2.2 Object Classes

The dataset contains 15 object categories commonly found in aerial imagery:

| Class ID | Class Name         | Description                   |
| -------- | ------------------ | ----------------------------- |
| 0        | Background         | Non-object regions            |
| 1        | Ship               | Maritime vessels              |
| 2        | Storage Tank       | Industrial storage containers |
| 3        | Baseball Diamond   | Sports facilities             |
| 4        | Tennis Court       | Sports facilities             |
| 5        | Basketball Court   | Sports facilities             |
| 6        | Ground Track Field | Athletic tracks               |
| 7        | Bridge             | Infrastructure                |
| 8        | Large Vehicle      | Trucks, buses                 |
| 9        | Small Vehicle      | Cars, motorcycles             |
| 10       | Helicopter         | Aircraft                      |
| 11       | Swimming Pool      | Recreational facilities       |
| 12       | Roundabout         | Road infrastructure           |
| 13       | Soccer Ball Field  | Sports facilities             |
| 14       | Plane              | Aircraft                      |
| 15       | Harbor             | Maritime infrastructure       |

### 2.3 Dataset Statistics

<!-- TODO: Add dataset analysis plots from 05_dataset_analysis.ipynb -->

**Class Distribution:**

![Class Distribution Placeholder](docs/images/class_distribution.png)
_Figure 2.1: Distribution of instances across classes in the training set_

**Objects per Image Distribution:**

![Objects per Image Placeholder](docs/images/objects_per_image.png)
_Figure 2.2: Histogram of number of objects per image_

**Bounding Box Size Distribution:**

![Box Size Distribution Placeholder](docs/images/box_size_distribution.png)
_Figure 2.3: Distribution of bounding box sizes (small objects dominate)_

### 2.4 Sample Images

<!-- TODO: Add sample images from the dataset -->

![Sample Image 1 Placeholder](docs/images/sample_1.png)
_Figure 2.4: Sample image with ground truth annotations showing ships and vehicles_

![Sample Image 2 Placeholder](docs/images/sample_2.png)
_Figure 2.5: Sample image showing storage tanks and large vehicles_

![Sample Image 3 Placeholder](docs/images/sample_3.png)
_Figure 2.6: Sample image with planes at an airport_

### 2.5 Dataset Preprocessing

The original high-resolution images are preprocessed into patches:

1. **Patching**: Original images are split into 800×800 patches with overlap
2. **Filtering**:
   - Images with >400 bounding boxes are excluded (memory optimization)
   - Empty images are limited to 30% of the dataset (training stability)
3. **Train/Val Split**: Standard DOTA-based split preserved

```python
# Dataset filtering configuration
max_boxes_per_image: 400  # Filter outliers with extreme box counts
max_empty_fraction: 0.3   # Control empty image ratio
```

---

## 3. Problem Description

### 3.1 Task Definition

**Instance Segmentation** is the task of detecting individual objects in an image and generating a pixel-level mask for each detected instance. Unlike semantic segmentation, which assigns a class to each pixel, instance segmentation distinguishes between different instances of the same class.

For each detected object, the model outputs:

- **Bounding Box**: `(x1, y1, x2, y2)` coordinates
- **Class Label**: One of 16 classes (including background)
- **Confidence Score**: Probability that the detection is correct
- **Instance Mask**: Binary mask indicating which pixels belong to the object

### 3.2 Challenges in Aerial Imagery

Aerial/satellite instance segmentation presents unique challenges:

| Challenge               | Description                                      | Our Solution                                    |
| ----------------------- | ------------------------------------------------ | ----------------------------------------------- |
| **Small Objects**       | Many objects (vehicles, ships) appear very small | Custom anchor sizes optimized for small objects |
| **Dense Scenes**        | Images can contain hundreds of objects           | Dataset filtering, efficient batch processing   |
| **Scale Variation**     | Objects vary from tiny cars to large fields      | Multi-scale FPN with 4 levels                   |
| **Class Imbalance**     | Vehicle classes dominate                         | Focal loss consideration (future work)          |
| **Similar Appearances** | Different classes can look similar from above    | CBAM attention for discriminative features      |
| **Partial Occlusion**   | Objects can be partially visible                 | RoI-based approach handles partial views        |

### 3.3 Evaluation Criteria

The primary evaluation metric is **mAP@0.5** (mean Average Precision at IoU threshold 0.5), following COCO evaluation protocols.

---

## 4. Model Architecture

### 4.1 Overall Architecture

Our model is a custom Mask R-CNN with the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Custom Mask R-CNN                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Input     │    │  Backbone   │    │        FPN          │ │
│  │   Image     │───▶│ EfficientNet│───▶│  + CBAM Attention   │ │
│  │  800×800    │    │  + CBAM     │    │                     │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                     ┌──────────────────────────────┘            │
│                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Region Proposal Network (RPN)             │   │
│  │   Custom Anchor Sizes: (8,16), (16,32), (32,64), (64,128)│   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RoI Heads                             │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌────────────────┐ │   │
│  │  │ Box Head    │   │ Box         │   │ Mask Head      │ │   │
│  │  │ (Custom FC) │   │ Predictor   │   │ (4 Conv+Deconv)│ │   │
│  │  └─────────────┘   └─────────────┘   └────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Outputs: Boxes, Labels, Scores, Masks                         │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.1: High-level architecture of Custom Mask R-CNN_

### 4.2 Backbone: EfficientNet-B0 with CBAM

The backbone extracts hierarchical features from input images using EfficientNet-B0 pre-trained on ImageNet, enhanced with CBAM (Convolutional Block Attention Module) at each scale.

```
┌─────────────────────────────────────────────────────────────────┐
│              EfficientNet-B0 Backbone with CBAM                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image (3, 800, 800)                                      │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  Stem Block  │  Conv 3×3, stride 2                          │
│  │  32 channels │                                              │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌────────┐                                │
│  │  MBConv ×1   │───▶│  CBAM  │──▶ C2 (40 ch, stride 4)       │
│  │  40 channels │    │   40   │                                │
│  └──────┬───────┘    └────────┘                                │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌────────┐                                │
│  │  MBConv ×2   │───▶│  CBAM  │──▶ C3 (80 ch, stride 8)       │
│  │  80 channels │    │   80   │                                │
│  └──────┬───────┘    └────────┘                                │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌────────┐                                │
│  │  MBConv ×3   │───▶│  CBAM  │──▶ C4 (112 ch, stride 16)     │
│  │  112 channels│    │  112   │                                │
│  └──────┬───────┘    └────────┘                                │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌────────┐                                │
│  │  MBConv ×4   │───▶│  CBAM  │──▶ C5 (320 ch, stride 32)     │
│  │  320 channels│    │  320   │                                │
│  └──────────────┘    └────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.2: Backbone architecture with CBAM attention at each stage_

### 4.3 CBAM (Convolutional Block Attention Module)

CBAM is a lightweight attention module that sequentially applies channel and spatial attention:

```
┌─────────────────────────────────────────────────────────────────┐
│                          CBAM Module                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Feature Map F (C × H × W)                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │            Channel Attention Module                  │       │
│  │  ┌─────────────┐    ┌─────────────┐                 │       │
│  │  │  AvgPool    │    │  MaxPool    │                 │       │
│  │  │  Global     │    │  Global     │                 │       │
│  │  └──────┬──────┘    └──────┬──────┘                 │       │
│  │         │                   │                        │       │
│  │         ▼                   ▼                        │       │
│  │  ┌─────────────────────────────────────────┐        │       │
│  │  │   Shared MLP (C → C/r → C)              │        │       │
│  │  └──────────────────┬──────────────────────┘        │       │
│  │                     │                                │       │
│  │              ┌──────┴──────┐                        │       │
│  │              │   Add + σ   │                        │       │
│  │              └──────┬──────┘                        │       │
│  └─────────────────────┼───────────────────────────────┘       │
│                        │                                        │
│         F' = F ⊗ Mc    │  (element-wise multiplication)        │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────────┐       │
│  │            Spatial Attention Module                  │       │
│  │  ┌─────────────┐    ┌─────────────┐                 │       │
│  │  │  AvgPool    │    │  MaxPool    │                 │       │
│  │  │  Channel    │    │  Channel    │                 │       │
│  │  └──────┬──────┘    └──────┬──────┘                 │       │
│  │         │                   │                        │       │
│  │         ▼                   ▼                        │       │
│  │  ┌─────────────────────────────────────────┐        │       │
│  │  │   Concat + Conv 7×7 + σ                 │        │       │
│  │  └──────────────────┬──────────────────────┘        │       │
│  └─────────────────────┼───────────────────────────────┘       │
│                        │                                        │
│         F'' = F' ⊗ Ms  │  (element-wise multiplication)        │
│                        ▼                                        │
│                  Output Feature Map                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.3: CBAM attention module structure (reduction ratio r=16)_

### 4.4 Feature Pyramid Network (FPN)

The FPN creates a multi-scale feature pyramid for detecting objects at different sizes:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Attention FPN Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Backbone Features              Lateral Conv      FPN Outputs   │
│                                                                 │
│  C5 (320ch, 1/32) ──────────▶ 1×1 Conv ─────────▶ P5 (256ch)   │
│                                    │         │                  │
│                                    ▼         ▼                  │
│  C4 (112ch, 1/16) ──────────▶ 1×1 Conv ─▶ + ─▶ P4 (256ch)     │
│                                    │     ↑   │                  │
│                                    ▼     │   ▼                  │
│  C3 (80ch, 1/8)  ──────────▶ 1×1 Conv ─▶ + ─▶ P3 (256ch)      │
│                                    │     ↑   │                  │
│                                    ▼     │   ▼                  │
│  C2 (40ch, 1/4)  ──────────▶ 1×1 Conv ─▶ + ─▶ P2 (256ch)      │
│                                          ↑                      │
│                                      2× Upsample                │
│                                                                 │
│  Each FPN level has:                                           │
│  - 3×3 Conv + BatchNorm + ReLU (smoothing)                     │
│  - CBAM attention module                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.4: FPN with top-down pathway and CBAM attention_

### 4.5 Region Proposal Network (RPN)

The RPN generates object proposals using custom anchor configurations optimized for aerial imagery:

**Anchor Configuration:**
| FPN Level | Anchor Sizes | Aspect Ratios | Effective Receptive Field |
|-----------|--------------|---------------|---------------------------|
| P2 | 8, 16 | 0.5, 1.0, 2.0 | Small objects (vehicles) |
| P3 | 16, 32 | 0.5, 1.0, 2.0 | Small-medium objects |
| P4 | 32, 64 | 0.5, 1.0, 2.0 | Medium objects |
| P5 | 64, 128 | 0.5, 1.0, 2.0 | Large objects (fields) |

### 4.6 RoI Heads

#### 4.6.1 Custom Box Head

```
┌─────────────────────────────────────────────────────────────────┐
│                    Custom Box Feature Extractor                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RoI Pooled Features (256 × 7 × 7 = 12,544)                    │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │   Flatten    │                                              │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  FC Layer 1  │  12,544 → 1,024 + ReLU                       │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  Dropout 0.3 │                                              │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  FC Layer 2  │  1,024 → 1,024 + ReLU                        │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  Output: Box Representations (1,024)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.5: Custom box feature extractor architecture_

#### 4.6.2 Custom Box Predictor

```
┌─────────────────────────────────────────────────────────────────┐
│                    Custom Box Predictor                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Box Representations (1,024)                                    │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  FC Layer    │  1,024 → 512 + ReLU                          │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  Dropout 0.3 │                                              │
│  └──────┬───────┘                                              │
│         │                                                       │
│    ┌────┴────┐                                                 │
│    │         │                                                 │
│    ▼         ▼                                                 │
│  ┌─────┐  ┌──────┐                                             │
│  │ Cls │  │ BBox │                                             │
│  │ 16  │  │ 64   │  (16 classes × 4 coordinates)              │
│  └─────┘  └──────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.6: Custom box predictor with classification and regression heads_

#### 4.6.3 Custom Mask Head

```
┌─────────────────────────────────────────────────────────────────┐
│                      Custom Mask Head                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RoI Pooled Features (256 × 14 × 14)                           │
│         │                                                       │
│    ┌────┴────┐  (residual connection)                          │
│    │         │                                                 │
│    ▼         │                                                 │
│  ┌──────────┐│                                                 │
│  │ Conv 3×3 ││  256 → 256 + BN + ReLU                          │
│  └────┬─────┘│                                                 │
│       │      │                                                 │
│       ▼      │                                                 │
│  ┌──────────┐│                                                 │
│  │ Conv 3×3 ││  256 → 256 + BN + ReLU                          │
│  └────┬─────┘│                                                 │
│       │      │                                                 │
│       ▼      │                                                 │
│  ┌──────────┐│                                                 │
│  │ Conv 3×3 ││  256 → 256 + BN + ReLU                          │
│  └────┬─────┘│                                                 │
│       │      │                                                 │
│       ▼      │                                                 │
│  ┌──────────┐│                                                 │
│  │ Conv 3×3 ││  256 → 256 + BN + ReLU                          │
│  └────┬─────┘│                                                 │
│       │      │                                                 │
│       ▼      │                                                 │
│    ┌──┴──┐   │                                                 │
│    │ Add │◀──┘  1×1 Conv for residual                          │
│    └──┬──┘                                                     │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────┐                                              │
│  │ ConvTranspose│  256 → 256, 2×2, stride 2 + BN + ReLU       │
│  │  (Upsample)  │                                              │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  Conv 1×1    │  256 → 16 (num_classes)                      │
│  │  Mask Logits │                                              │
│  └──────────────┘                                              │
│                                                                 │
│  Output: (N, 16, 28, 28) mask logits per RoI                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_Figure 4.7: Custom mask head with 4 convolutional layers and residual connection_

### 4.7 Custom Layers Summary

Our implementation contains **>50% custom layers** as required:

| Component                 | Custom Layers                      | Description                        |
| ------------------------- | ---------------------------------- | ---------------------------------- |
| **CBAM Modules**          | ChannelAttention, SpatialAttention | Attention mechanisms               |
| **Backbone CBAM**         | 4 CBAM modules                     | Added to each backbone stage       |
| **FPN CBAM**              | 4 CBAM modules                     | Added to each FPN level            |
| **Box Feature Extractor** | 2 FC layers + Dropout              | Custom representation learning     |
| **Box Predictor**         | FC + Cls + BBox layers             | Additional layer before prediction |
| **Mask Head**             | 4 Conv + Residual + Deconv         | Extended mask prediction           |

---

## 5. Model Analysis

### 5.1 Model Size

| Metric                   | Value       |
| ------------------------ | ----------- |
| **Total Parameters**     | ~XX,XXX,XXX |
| **Trainable Parameters** | ~XX,XXX,XXX |
| **Model Size (FP32)**    | ~XXX MB     |
| **Model Size (FP16)**    | ~XXX MB     |

<!-- TODO: Fill in actual values from training -->

### 5.2 Parameter Breakdown by Component

| Component                  | Parameters | Percentage |
| -------------------------- | ---------- | ---------- |
| Backbone (EfficientNet-B0) | ~4.0M      | XX%        |
| CBAM Modules               | ~XXX K     | XX%        |
| FPN                        | ~XXX K     | XX%        |
| RPN                        | ~XXX K     | XX%        |
| Box Head                   | ~XXX M     | XX%        |
| Mask Head                  | ~XXX K     | XX%        |
| **Total**                  | ~XX M      | 100%       |

### 5.3 Memory Requirements

| Configuration                | GPU Memory |
| ---------------------------- | ---------- |
| Training (batch_size=8, AMP) | ~XX GB     |
| Training (batch_size=4, AMP) | ~XX GB     |
| Inference (single image)     | ~XX GB     |

### 5.4 FLOPs Analysis

<!-- TODO: Add FLOPs calculation -->

| Input Size | FLOPs      |
| ---------- | ---------- |
| 800×800    | ~XX GFLOPs |

---

## 6. Training Description

### 6.1 Training Commands

#### Option 1: Using the Training Script (Recommended)

```bash
# Basic training with default config
python train.py --config config.yaml --data-root /path/to/iSAID_patches

# Training with custom parameters
python train.py \
    --config config.yaml \
    --data-root /path/to/iSAID_patches \
    --output-dir ./checkpoints \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0003

# Training without W&B logging
python train.py --config config.yaml --data-root /path/to/data --no-wandb

# Training with LR finder
python train.py --config config.yaml --data-root /path/to/data --find-lr

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/last.pth
```

#### Option 2: Using Docker (Optional)

```bash
# Build the Docker image
docker-compose build

# Create .env file
cp .env.example .env
# Edit .env with your WANDB_API_KEY and DATA_PATH

# Run training
docker-compose run train

# Run with custom parameters
docker-compose run train --epochs 10 --batch-size 4

# Interactive development shell
docker-compose run dev
```

### 6.2 Configuration File

The training is configured via `config.yaml`:

```yaml
# Dataset
data:
  root_dir: "iSAID_patches"
  num_classes: 16
  image_size: 800
  max_boxes_per_image: 400
  max_empty_fraction: 0.3

# Model
model:
  backbone: "efficientnet_b0"
  pretrained_backbone: true

# Training
training:
  epochs: 25
  batch_size: 8
  learning_rate: 0.0003
  weight_decay: 0.01
  amp: true
```

### 6.3 Training Pipeline

1. **Data Loading**: Images and annotations loaded with augmentations
2. **Forward Pass**: Image → Backbone → FPN → RPN → RoI Heads
3. **Loss Computation**: Multi-task loss (classification + regression + mask)
4. **Backward Pass**: Gradient computation with gradient clipping (max_norm=1.0)
5. **Optimization**: AdamW with differential learning rates
6. **Validation**: Periodic mAP computation on validation set
7. **Checkpointing**: Save best model by validation loss and mAP

### 6.4 Data Augmentation

Training augmentations (applied with Albumentations):

| Augmentation    | Parameters                   | Purpose                       |
| --------------- | ---------------------------- | ----------------------------- |
| Horizontal Flip | p=0.5                        | Increase viewpoint invariance |
| Color Jitter    | brightness=0.2, contrast=0.2 | Handle lighting variations    |
| Random Resize   | scale=(0.8, 1.2)             | Multi-scale training          |

---

## 7. Metrics and Evaluation

### 7.1 Loss Functions

The total loss is a combination of four components:

$$\mathcal{L}_{total} = \mathcal{L}_{rpn\_cls} + \mathcal{L}_{rpn\_reg} + \mathcal{L}_{roi\_cls} + \mathcal{L}_{roi\_reg} + \mathcal{L}_{mask}$$

| Loss Component         | Type                 | Description                |
| ---------------------- | -------------------- | -------------------------- |
| **RPN Classification** | Binary Cross-Entropy | Object vs. background      |
| **RPN Regression**     | Smooth L1            | Anchor box refinement      |
| **RoI Classification** | Cross-Entropy        | Multi-class classification |
| **RoI Regression**     | Smooth L1            | Bounding box refinement    |
| **Mask Loss**          | Binary Cross-Entropy | Per-pixel mask prediction  |

### 7.2 Evaluation Metrics

#### 7.2.1 mAP (mean Average Precision)

The primary metric, computed as:

1. For each class, compute AP using 11-point interpolation
2. Average across all classes

$$mAP = \frac{1}{N_{classes}} \sum_{c=1}^{N_{classes}} AP_c$$

#### 7.2.2 IoU (Intersection over Union)

$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

Where A is the predicted box/mask and B is the ground truth.

#### 7.2.3 Mean IoU

Average IoU across all matched predictions and ground truths.

### 7.3 Evaluation Protocol

- **IoU Threshold**: 0.5 (following COCO protocol)
- **Score Threshold**: 0.5 for positive detections
- **Max Detections**: 100 per image
- **Evaluation Samples**: 200 images (for computational efficiency)

---

## 8. Hyperparameters

### 8.1 Hyperparameter Table

| Hyperparameter            | Value | Rationale                                                    |
| ------------------------- | ----- | ------------------------------------------------------------ |
| **Learning Rate**         | 3e-4  | Found via LR range test; standard for AdamW                  |
| **Batch Size**            | 8     | Maximum fitting in GPU memory with AMP                       |
| **Epochs**                | 25    | Sufficient for convergence on this dataset                   |
| **Weight Decay**          | 0.01  | Standard regularization for AdamW                            |
| **Image Size**            | 800   | Balances resolution and computation; standard for Mask R-CNN |
| **RoI LR Multiplier**     | 0.25  | Lower LR for RoI heads prevents overfitting                  |
| **Gradient Clip Norm**    | 1.0   | Stabilizes training, prevents exploding gradients            |
| **AMP (Mixed Precision)** | True  | 2× memory savings, faster training                           |
| **Dropout**               | 0.3   | Regularization in box/mask heads                             |
| **CBAM Reduction**        | 16    | Balance between capacity and computation                     |

### 8.2 Anchor Hyperparameters

| Parameter         | Value                              | Rationale                                           |
| ----------------- | ---------------------------------- | --------------------------------------------------- |
| **Anchor Sizes**  | (8,16), (16,32), (32,64), (64,128) | Optimized for small aerial objects                  |
| **Aspect Ratios** | 0.5, 1.0, 2.0                      | Cover horizontal, square, and vertical objects      |
| **RPN FG IoU**    | 0.5                                | Lowered from 0.7 for better recall on small objects |
| **RPN BG IoU**    | 0.3                                | Standard threshold                                  |
| **RPN NMS**       | 0.7                                | Standard non-maximum suppression                    |

### 8.3 Scheduler Configuration

**ReduceLROnPlateau:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Mode | max | Monitoring mAP (higher is better) |
| Factor | 0.5 | Halve LR on plateau |
| Patience | 3 | Wait 3 epochs before reducing |
| Min LR | 1e-6 | Lower bound for learning rate |

---

## 9. Results and Plots

### 9.1 Training Curves

<!-- TODO: Add training curves from W&B or local training -->

#### 9.1.1 Loss Curves

![Training and Validation Loss](docs/images/loss_curves.png)
_Figure 9.1: Training and validation loss over epochs_

#### 9.1.2 mAP Curves

![mAP Curves](docs/images/map_curves.png)
_Figure 9.2: Training and validation mAP@0.5 over epochs_

#### 9.1.3 Learning Rate Schedule

![Learning Rate](docs/images/lr_schedule.png)
_Figure 9.3: Learning rate schedule with ReduceLROnPlateau_

#### 9.1.4 Gradient Norms

![Gradient Norms](docs/images/gradient_norms.png)
_Figure 9.4: Gradient norm during training (stability indicator)_

### 9.2 Final Results

| Metric         | Train | Validation |
| -------------- | ----- | ---------- |
| **Total Loss** | X.XXX | X.XXX      |
| **mAP@0.5**    | X.XXX | X.XXX      |
| **Mean IoU**   | -     | X.XXX      |

<!-- TODO: Fill in actual values -->

### 9.3 Per-Class Results

| Class              | AP@0.5 | Notes |
| ------------------ | ------ | ----- |
| Ship               | X.XX   |       |
| Storage Tank       | X.XX   |       |
| Baseball Diamond   | X.XX   |       |
| Tennis Court       | X.XX   |       |
| Basketball Court   | X.XX   |       |
| Ground Track Field | X.XX   |       |
| Bridge             | X.XX   |       |
| Large Vehicle      | X.XX   |       |
| Small Vehicle      | X.XX   |       |
| Helicopter         | X.XX   |       |
| Swimming Pool      | X.XX   |       |
| Roundabout         | X.XX   |       |
| Soccer Ball Field  | X.XX   |       |
| Plane              | X.XX   |       |
| Harbor             | X.XX   |       |

### 9.4 Qualitative Results

<!-- TODO: Add prediction visualizations -->

![Prediction Sample 1](docs/images/prediction_1.png)
_Figure 9.5: Model predictions on validation sample 1_

![Prediction Sample 2](docs/images/prediction_2.png)
_Figure 9.6: Model predictions on validation sample 2_

![Prediction Sample 3](docs/images/prediction_3.png)
_Figure 9.7: Model predictions on validation sample 3 (challenging dense scene)_

---

## 10. Model Comparison

### 10.1 Backbone Comparison

<!-- TODO: Add results from backbone comparison experiments -->

| Backbone               | Parameters | mAP@0.5 | Training Time | Notes              |
| ---------------------- | ---------- | ------- | ------------- | ------------------ |
| EfficientNet-B0 + CBAM | XX M       | X.XX    | XX hrs        | Default choice     |
| ResNet-50 + CBAM       | XX M       | X.XX    | XX hrs        | Heavier but stable |
| MobileNetV3 + CBAM     | XX M       | X.XX    | XX hrs        | Lightweight option |

### 10.2 Ablation Studies

| Configuration                                 | mAP@0.5 | Δ mAP    |
| --------------------------------------------- | ------- | -------- |
| Full Model (EfficientNet + CBAM + Custom RoI) | X.XX    | baseline |
| Without CBAM                                  | X.XX    | -X.XX    |
| Without Custom RoI Heads                      | X.XX    | -X.XX    |
| Without Custom Anchors                        | X.XX    | -X.XX    |
| With Default Anchors (32, 64, 128, 256, 512)  | X.XX    | -X.XX    |

### 10.3 Anchor Optimization Results

<!-- TODO: Add anchor optimization results from 02_anchoropt.ipynb -->

| Anchor Configuration | IoU Coverage | mAP@0.5 |
| -------------------- | ------------ | ------- |
| Default (32-512)     | X.XX         | X.XX    |
| Optimized (8-128)    | X.XX         | X.XX    |

---

## 11. Challenges and Solutions

### 11.1 Memory Management

**Challenge:** Training on high-resolution images with dense annotations causes GPU memory overflow.

**Solutions:**

- Mixed precision training (AMP) reduces memory by ~50%
- Dataset filtering removes outlier images with >400 boxes
- Gradient checkpointing (future work)
- Efficient data loading with `pin_memory=True`

### 11.2 Small Object Detection

**Challenge:** Many objects in aerial images are very small (<32 pixels).

**Solutions:**

- Custom anchor sizes starting from 8 pixels
- Lower RPN foreground IoU threshold (0.5 instead of 0.7)
- Multi-scale FPN with P2 level for fine details
- CBAM attention for better feature refinement

### 11.3 Class Imbalance

**Challenge:** Vehicle classes have many more instances than rare classes.

**Solutions:**

- Stratified sampling in RoI heads (positive_fraction=0.25)
- Balanced batch composition
- Future work: Focal loss implementation

### 11.4 Training Stability

**Challenge:** Loss spikes and NaN values during training.

**Solutions:**

- Gradient clipping (max_norm=1.0)
- NaN/Inf detection and batch skipping
- Differential learning rates (lower for RoI heads)
- AdamW optimizer with weight decay

### 11.5 Overfitting

**Challenge:** Gap between training and validation mAP.

**Solutions:**

- Dropout (0.3) in box and mask heads
- Data augmentation (flip, color jitter)
- Lower learning rate for RoI heads
- Early stopping based on mAP gap (optional)

---

## 12. Runtime Environment

### 12.1 Hardware

| Component      | Specification      |
| -------------- | ------------------ |
| **GPU**        | NVIDIA [GPU Model] |
| **GPU Memory** | XX GB              |
| **CPU**        | [CPU Model]        |
| **RAM**        | XX GB              |
| **Storage**    | [Storage Type]     |

### 12.2 Software

| Component   | Version               |
| ----------- | --------------------- |
| **OS**      | [OS Name and Version] |
| **Python**  | 3.10+                 |
| **PyTorch** | 2.0.0+                |
| **CUDA**    | 11.8                  |
| **cuDNN**   | 8.x                   |

### 12.3 Docker Environment (Optional)

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
```

See `Dockerfile` and `docker-compose.yml` for full configuration.

---

## 13. Training and Inference Time

### 13.1 Training Time

| Configuration     | Time per Epoch | Total (25 epochs) |
| ----------------- | -------------- | ----------------- |
| Batch Size 8, AMP | ~XX min        | ~XX hrs           |
| Batch Size 4, AMP | ~XX min        | ~XX hrs           |

### 13.2 Inference Time

| Configuration | Time per Image | FPS |
| ------------- | -------------- | --- |
| 800×800, GPU  | ~XX ms         | ~XX |
| 800×800, CPU  | ~XX ms         | ~XX |

### 13.3 mAP Computation Time

| Samples         | Time    |
| --------------- | ------- |
| 200             | ~XX min |
| Full Validation | ~XX min |

---

## 14. Libraries and Tools

### 14.1 Core Dependencies

See `requirements.txt` for complete list:

```
# PyTorch (CUDA 11.8)
torch>=2.0.0
torchvision>=0.15.0

# Core dependencies
numpy>=1.24.0
pillow>=9.0.0
opencv-python>=4.7.0
pyyaml>=6.0

# Image augmentation
albumentations>=1.3.0

# Visualization
matplotlib>=3.7.0
scipy>=1.10.0

# Deep learning utilities
timm>=0.9.0
tqdm>=4.65.0

# Hyperparameter optimization
optuna>=3.0.0

# Experiment tracking
wandb>=0.15.0
```

### 14.2 Tools

| Tool                 | Purpose                               |
| -------------------- | ------------------------------------- |
| **Weights & Biases** | Experiment tracking, visualization    |
| **Optuna**           | Hyperparameter optimization (anchors) |
| **Albumentations**   | Data augmentation                     |
| **Docker**           | Reproducible environment (optional)   |
| **Git**              | Version control                       |

---

## 15. Bibliography

### 15.1 Dataset

1. Waqas Zamir, S., Arora, A., Gupta, A., Khan, S., Sun, G., Shahbaz Khan, F., ... & Shao, L. (2019). **iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images**. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 28-37). [Paper Link](https://arxiv.org/abs/1905.12886)

2. Xia, G. S., Bai, X., Ding, J., Zhu, Z., Belongie, S., Luo, J., ... & Zhang, L. (2018). **DOTA: A Large-scale Dataset for Object Detection in Aerial Images**. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 3974-3983). [Paper Link](https://arxiv.org/abs/1711.10398)

### 15.2 Model Architecture

3. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). **Mask R-CNN**. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2961-2969). [Paper Link](https://arxiv.org/abs/1703.06870)

4. Tan, M., & Le, Q. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. In International Conference on Machine Learning (pp. 6105-6114). [Paper Link](https://arxiv.org/abs/1905.11946)

5. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). **CBAM: Convolutional Block Attention Module**. In Proceedings of the European Conference on Computer Vision (pp. 3-19). [Paper Link](https://arxiv.org/abs/1807.06521)

6. Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). **Feature Pyramid Networks for Object Detection**. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2117-2125). [Paper Link](https://arxiv.org/abs/1612.03144)

### 15.3 Training Techniques

7. Smith, L. N. (2017). **Cyclical Learning Rates for Training Neural Networks**. In 2017 IEEE Winter Conference on Applications of Computer Vision (pp. 464-472). [Paper Link](https://arxiv.org/abs/1506.01186)

8. Loshchilov, I., & Hutter, F. (2017). **Decoupled Weight Decay Regularization**. arXiv preprint arXiv:1711.05101. [Paper Link](https://arxiv.org/abs/1711.05101)

### 15.4 Online Resources

9. PyTorch Vision Detection Documentation: [https://pytorch.org/vision/stable/models.html#object-detection](https://pytorch.org/vision/stable/models.html#object-detection)

10. Weights & Biases Documentation: [https://docs.wandb.ai/](https://docs.wandb.ai/)

11. Albumentations Documentation: [https://albumentations.ai/docs/](https://albumentations.ai/docs/)

---

## 16. Evaluation Checklist

| #         | Requirement                                    | Status  | Points    |
| --------- | ---------------------------------------------- | ------- | --------- |
| 1         | Description of the dataset with image examples | ✅ / ⬜ | X         |
| 2         | Description of the problem                     | ✅ / ⬜ | X         |
| 3         | Description of architectures with diagrams     | ✅ / ⬜ | X         |
| 4         | Model analysis (size, parameters)              | ✅ / ⬜ | X         |
| 5         | Training description with commands             | ✅ / ⬜ | X         |
| 6         | Metrics, loss, and evaluation description      | ✅ / ⬜ | X         |
| 7         | Training/validation loss and metrics plots     | ⬜      | X         |
| 8         | Hyperparameters with explanations              | ✅ / ⬜ | X         |
| 9         | Model comparison                               | ⬜      | X         |
| 10        | Libraries and tools (requirements.txt)         | ✅      | X         |
| 11        | Runtime environment description                | ✅ / ⬜ | X         |
| 12        | Training and inference time                    | ⬜      | X         |
| 13        | Bibliography                                   | ✅ / ⬜ | X         |
| 14        | Git repository link                            | ⬜      | X         |
| **Total** |                                                |         | **X / Y** |

---

## Appendix A: Project Structure

```
isaid-instance-segmentation/
├── config.yaml                 # Main configuration file
├── train.py                    # Training script (CLI)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
├── REPORT.md                   # This report
├── README.md                   # Quick start guide
│
├── datasets/
│   ├── __init__.py
│   └── isaid_dataset.py        # Dataset loader with filtering
│
├── models/
│   ├── __init__.py
│   ├── backbone.py             # EfficientNet + CBAM backbone
│   ├── maskrcnn_model.py       # Custom Mask R-CNN
│   └── roi_heads.py            # Custom RoI heads
│
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training loop with W&B integration
│   ├── transforms.py           # Data augmentation
│   ├── anchor_optimizer.py     # Optuna-based anchor optimization
│   └── wandb_logger.py         # W&B logging utilities
│
├── visualization/
│   ├── __init__.py
│   └── gradcam_pipeline.py     # GradCAM visualization
│
├── utils/
│   ├── __init__.py
│   └── overfit_test.py         # Debugging utilities
│
└── notebooks/
    ├── 00_setup.ipynb          # Environment setup
    ├── 01_training.ipynb       # Basic training notebook
    ├── 02_anchoropt.ipynb      # Anchor optimization
    ├── 03_training_wandb.ipynb # Training with W&B
    ├── 04_training_kaggle_backbone_choice.ipynb
    ├── 05_dataset_analysis.ipynb
    └── 06_gradcam_visualization.ipynb
```

---

## Appendix B: W&B Dashboard

<!-- TODO: Add W&B dashboard screenshot -->

![W&B Dashboard](docs/images/wandb_dashboard.png)
_Figure B.1: Weights & Biases experiment tracking dashboard_

---

_Report generated: January 2026_
