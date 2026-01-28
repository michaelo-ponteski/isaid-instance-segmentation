#!/usr/bin/env python3
"""
Training script for iSAID Instance Segmentation with Mask R-CNN.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --data-root /path/to/data
    python train.py --config config.yaml --no-wandb
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from typing import Optional

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.isaid_dataset import iSAIDDataset
from models.maskrcnn_model import CustomMaskRCNN
from training.trainer import Trainer, create_datasets
from training.transforms import get_transforms


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on iSAID dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )

    # Override options
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override checkpoint save directory from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="W&B API key (can also be set via WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Run learning rate finder before training",
    )

    return parser.parse_args()


def setup_wandb(config: dict, api_key: Optional[str] = None):
    """Setup W&B authentication."""
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    # Check if API key is available
    if "WANDB_API_KEY" not in os.environ:
        print("Warning: WANDB_API_KEY not set. W&B logging may fail.")
        print("Set it via --wandb-key argument or WANDB_API_KEY environment variable.")


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Apply command line overrides
    data_root = args.data_root or config["data"]["root_dir"]
    output_dir = args.output_dir or config["checkpoint"]["save_dir"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Device setup
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\n{'='*60}")
    print("iSAID Instance Segmentation Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")

    # Setup CUDA memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Create datasets
    print("Loading datasets...")
    train_dataset, val_dataset = create_datasets(
        data_root=data_root,
        image_size=config["data"]["image_size"],
        subset_fraction=config["data"].get("subset_fraction", 1.0),
        max_boxes_per_image=config["data"].get("max_boxes_per_image", 400),
        max_empty_fraction=config["data"].get("max_empty_fraction", 0.3),
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    anchor_sizes = tuple(tuple(s) for s in config["anchors"]["sizes"])
    aspect_ratios = tuple(config["anchors"]["aspect_ratios"])
    rpn_aspect_ratios = tuple([aspect_ratios] * len(anchor_sizes))

    model = CustomMaskRCNN(
        num_classes=config["data"]["num_classes"],
        pretrained_backbone=config["model"]["pretrained_backbone"],
        rpn_anchor_sizes=anchor_sizes,
        rpn_aspect_ratios=rpn_aspect_ratios,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # W&B configuration
    wandb_config = config.get("wandb", {})
    use_wandb = wandb_config.get("enabled", True) and not args.no_wandb

    if use_wandb:
        setup_wandb(config, args.wandb_key)

    # Build hyperparameters dict for W&B
    hyperparameters = {
        # Dataset
        "data_root": data_root,
        "num_classes": config["data"]["num_classes"],
        "image_size": config["data"]["image_size"],
        # Training
        "batch_size": batch_size,
        "val_batch_size": config["training"].get("val_batch_size", batch_size),
        "num_epochs": epochs,
        "learning_rate": lr,
        "weight_decay": config["training"].get("weight_decay", 0.01),
        # Model
        "backbone": config["model"].get("backbone", "efficientnet_b0"),
        "pretrained_backbone": config["model"]["pretrained_backbone"],
        # Anchors
        "anchor_sizes": anchor_sizes,
        "aspect_ratios": aspect_ratios,
    }

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        batch_size=batch_size,
        val_batch_size=config["training"].get("val_batch_size", batch_size),
        lr=lr,
        device=device,
        use_amp=config["training"].get("amp", True),
        num_workers=config["training"].get("num_workers", 4),
        # W&B configuration
        wandb_project=wandb_config.get("project") if use_wandb else None,
        wandb_entity=wandb_config.get("entity"),
        wandb_tags=wandb_config.get("tags", []),
        wandb_notes=wandb_config.get("notes", ""),
        wandb_log_freq=wandb_config.get("log_freq", 20),
        wandb_num_val_images=wandb_config.get("num_val_images", 4),
        wandb_conf_threshold=wandb_config.get("conf_threshold", 0.5),
        hyperparameters=hyperparameters,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    try:
        history = trainer.fit(
            epochs=epochs,
            save_dir=output_dir,
            find_lr_first=args.find_lr,
            compute_metrics_every=config["training"].get("compute_metrics_every", 1),
            max_map_samples=config["training"].get("max_map_samples", 200),
            early_stop_gap_patience=config["training"].get("early_stop_gap_patience"),
            early_stop_gap_threshold=config["training"].get(
                "early_stop_gap_threshold", 0.01
            ),
        )

        # Print final results
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Final train loss: {history['train/loss'][-1]:.4f}")
        print(f"Final val loss: {history['val/loss'][-1]:.4f}")
        print(f"Best val mAP@0.5: {max(history['val/mAP@0.5']):.4f}")
        print(f"Checkpoints saved to: {output_dir}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Cleanup
        trainer.finish()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
