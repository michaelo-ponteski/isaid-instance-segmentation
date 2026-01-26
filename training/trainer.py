"""
Training loop for Mask R-CNN (memory-safe version) with optional W&B integration.
"""

import os
import time
import gc
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from models.maskrcnn_model import (
    CustomMaskRCNN,
    get_custom_maskrcnn,
    get_custom_maskrcnn_with_optimized_anchors,
)
from datasets.isaid_dataset import iSAIDDataset
from training.transforms import get_transforms

# Optional W&B import
try:
    from training.wandb_logger import (
        WandbLogger,
        WandbConfig,
        compute_gradient_norms,
        get_fixed_val_batch,
        ISAID_CLASS_LABELS,
    )

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    WandbConfig = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: (N, 4) tensor of boxes in xyxy format
        box2: (M, 4) tensor of boxes in xyxy format

    Returns:
        (N, M) tensor of IoU values
    """
    # Intersection coordinates
    x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    # Intersection area
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Union area
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


def compute_ap_single_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision for a single class at given IoU threshold.

    Uses the 11-point interpolation method for AP computation.

    Args:
        pred_boxes: (N, 4) predicted boxes
        pred_scores: (N,) confidence scores
        gt_boxes: (M, 4) ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        AP value (float)
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # Compute IoU matrix
    ious = compute_iou(pred_boxes, gt_boxes)

    # Track which GT boxes have been matched
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):
        # Find best matching GT box
        iou_max, gt_idx = ious[i].max(dim=0)

        if iou_max >= iou_threshold and not gt_matched[gt_idx]:
            tp[i] = 1
            gt_matched[gt_idx] = True
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 11-point interpolation for AP
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max().item() / 11

    return ap


def create_datasets(
    data_root: str,
    image_size: int = 800,
    subset_fraction: float = 1.0,
    # Dataset filtering parameters (applied before training)
    max_boxes_per_image: int = 400,
    max_empty_fraction: float = 0.3,
    filter_seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets with optional filtering.

    This is a helper function to create datasets once and reuse them
    across multiple training runs with different models/backbones.

    Filtering is applied BEFORE training to:
    - Remove outlier images with extreme box counts (prevents RAM/VRAM spikes)
    - Control the fraction of empty images (prevents empty-dominated batches)

    Args:
        data_root: Path to the dataset root directory
        image_size: Image size for resizing
        subset_fraction: Fraction of data to use (0.0 to 1.0)
        max_boxes_per_image: Max boxes per image. Images with more are excluded.
            Set to None to disable. Default: 400
        max_empty_fraction: Max fraction of empty images (0 boxes).
            Set to 1.0 to keep all. Default: 0.3 (30%)
        filter_seed: Random seed for deterministic empty image sampling.

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("Loading datasets...")
    train_dataset_full = iSAIDDataset(
        data_root,
        split="train",
        transforms=get_transforms(train=True),
        image_size=image_size,
        max_boxes_per_image=max_boxes_per_image,
        max_empty_fraction=max_empty_fraction,
        filter_seed=filter_seed,
    )
    val_dataset_full = iSAIDDataset(
        data_root,
        split="val",
        transforms=get_transforms(train=False),
        image_size=image_size,
        max_boxes_per_image=max_boxes_per_image,
        max_empty_fraction=max_empty_fraction,
        filter_seed=filter_seed,
    )

    # Apply subset if needed
    if subset_fraction < 1.0:
        train_size = int(len(train_dataset_full) * subset_fraction)
        val_size = int(len(val_dataset_full) * subset_fraction)
        train_size = max(1, train_size)
        val_size = max(1, val_size)

        train_dataset = Subset(train_dataset_full, range(train_size))
        val_dataset = Subset(val_dataset_full, range(val_size))
        print(
            f"Using {subset_fraction*100:.1f}% of data: {train_size} train, {val_size} val samples"
        )
    else:
        train_dataset = train_dataset_full
        val_dataset = val_dataset_full
        print(
            f"Using full dataset: {len(train_dataset_full)} train, {len(val_dataset_full)} val samples"
        )

    return train_dataset, val_dataset


class Trainer:
    """
    Trainer class for Mask R-CNN models with optional W&B integration.

    Can be initialized in two ways:
    1. With pre-created datasets and model (recommended for multiple runs)
    2. With data_root to create datasets internally (legacy mode)

    W&B Integration:
        Pass wandb_config to enable comprehensive W&B logging including:
        - Training/validation losses and metrics
        - Gradient norms for CBAM and RoI layers
        - Learning rate tracking
        - Validation image predictions visualization
        - Model checkpointing as artifacts

    Example usage with W&B:
        # Create datasets and model
        train_ds, val_ds = create_datasets("iSAID_patches", image_size=800)
        model = CustomMaskRCNN(num_classes=16)

        # Define hyperparameters (logged to W&B)
        hyperparams = {
            "learning_rate": 0.005,
            "batch_size": 2,
            "num_epochs": 50,
            "backbone": "efficientnet_b0",
        }

        # Create trainer with W&B
        trainer = Trainer(
            train_dataset=train_ds,
            val_dataset=val_ds,
            model=model,
            wandb_project="my-project",
            wandb_tags=["experiment-1"],
            hyperparameters=hyperparams,
        )
        trainer.fit(epochs=50)

    Example usage with external datasets and model:
        # Create datasets once
        train_ds, val_ds = create_datasets("iSAID_patches", image_size=800)

        # Create model with custom backbone
        model = CustomMaskRCNN(num_classes=16, backbone_with_fpn=my_backbone)

        # Train
        trainer = Trainer(
            train_dataset=train_ds,
            val_dataset=val_ds,
            model=model,
        )
        trainer.train(epochs=20)

        # Train another model with same datasets
        model2 = CustomMaskRCNN(num_classes=16, backbone_with_fpn=other_backbone)
        trainer2 = Trainer(
            train_dataset=train_ds,
            val_dataset=val_ds,
            model=model2,
        )
    """

    def __init__(
        self,
        # New interface: pass datasets and model directly
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        model: Optional[CustomMaskRCNN] = None,
        # Legacy interface: create datasets from data_root
        data_root: Optional[str] = None,
        num_classes: int = 16,
        # Training parameters
        batch_size: int = 2,
        val_batch_size: int = 1,
        lr: float = 3e-4,  # base_lr, RoI heads use lr * 0.25
        device: str = "cuda",
        use_amp: bool = True,
        num_workers: int = 2,
        # Legacy dataset creation parameters
        image_size: int = 800,
        subset_fraction: float = 1.0,
        # Legacy anchor optimization parameters
        optimize_anchors: bool = False,
        anchor_optimization_trials: int = 20,
        rpn_anchor_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
        rpn_aspect_ratios: Optional[Tuple[Tuple[float, ...], ...]] = None,
        anchor_cache_path: str = "optimized_anchors.pt",
        # W&B integration parameters
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_notes: str = "",
        wandb_log_freq: int = 20,
        wandb_num_val_images: int = 4,
        wandb_conf_threshold: float = 0.5,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Trainer.

        Args:
            train_dataset: Pre-created training dataset (recommended)
            val_dataset: Pre-created validation dataset (recommended)
            model: Pre-created CustomMaskRCNN model (recommended)
            data_root: Path to dataset (legacy mode - creates datasets internally)
            num_classes: Number of classes (used only if model is None)
            batch_size: Training batch size
            lr: Learning rate
            device: Device to train on ('cuda' or 'cpu')
            use_amp: Whether to use automatic mixed precision
            num_workers: Number of data loading workers
            image_size: Image size (used only if datasets are None)
            subset_fraction: Fraction of data to use (used only if datasets are None)
            optimize_anchors: Whether to run anchor optimization (legacy)
            anchor_optimization_trials: Number of Optuna trials (legacy)
            rpn_anchor_sizes: Custom anchor sizes (legacy)
            rpn_aspect_ratios: Custom aspect ratios (legacy)
            anchor_cache_path: Path to cache optimized anchors (legacy)
            wandb_project: W&B project name (enables W&B logging if set)
            wandb_entity: W&B entity/team name
            wandb_run_name: W&B run name (auto-generated if None)
            wandb_tags: List of tags for the W&B run
            wandb_notes: Notes for the W&B run
            wandb_log_freq: How often to log training metrics (every N steps)
            wandb_num_val_images: Number of validation images to visualize
            wandb_conf_threshold: Confidence threshold for prediction visualization
            hyperparameters: Dictionary of hyperparameters to log to W&B
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.val_batch_size = val_batch_size

        # W&B logger (initialized later if wandb_project is provided)
        self.wandb_logger = None  # Optional[WandbLogger]
        self.wandb_project = wandb_project
        self._global_step = 0
        self._val_image_indices = None  # Optional[List[int]]
        self._fixed_val_images = None  # Optional[List[torch.Tensor]]
        self._fixed_val_targets = None  # Optional[List[Dict]]

        # Handle datasets
        if train_dataset is not None and val_dataset is not None:
            # Use provided datasets
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            print(
                f"Using provided datasets: {len(train_dataset)} train, {len(val_dataset)} val samples"
            )
        elif data_root is not None:
            # Legacy mode: create datasets from data_root
            self.train_dataset, self.val_dataset = create_datasets(
                data_root=data_root,
                image_size=image_size,
                subset_fraction=subset_fraction,
            )
            self.data_root = data_root
        else:
            raise ValueError("Either provide (train_dataset, val_dataset) or data_root")

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        # Use num_workers=0 for validation to avoid multiprocessing issues in notebooks
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Handle model
        if model is not None:
            # Use provided model
            print("Using provided model")
            self.model = model
        elif data_root is not None:
            # Legacy mode: create model
            print("Creating model...")
            if optimize_anchors:
                print("Running anchor optimization with Optuna...")
                self.model = get_custom_maskrcnn_with_optimized_anchors(
                    num_classes=num_classes,
                    data_root=data_root,
                    n_trials=anchor_optimization_trials,
                    device=device,
                    image_size=image_size,
                    cache_path=anchor_cache_path,
                )
            elif rpn_anchor_sizes is not None or rpn_aspect_ratios is not None:
                print("Using custom anchor configuration...")
                self.model = get_custom_maskrcnn(
                    num_classes=num_classes,
                    rpn_anchor_sizes=rpn_anchor_sizes,
                    rpn_aspect_ratios=rpn_aspect_ratios,
                )
            else:
                print("Using default anchor configuration...")
                self.model = get_custom_maskrcnn(num_classes=num_classes)
        else:
            raise ValueError("Either provide a model or data_root to create one")

        self.model.to(self.device)

        # Setup optimizer with differential learning rates
        # RoI heads (box_head, mask_head, box_predictor, mask_predictor) use lower LR
        # to prevent overfitting on the smaller detection head
        roi_lr_alpha = 0.25  # RoI heads use base_lr * alpha
        roi_keywords = ['roi_heads', 'box_head', 'mask_head', 'box_predictor', 'mask_predictor']
        
        roi_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in roi_keywords):
                roi_params.append(param)
            else:
                base_params.append(param)
        
        param_groups = [
            {'params': base_params, 'lr': lr},
            {'params': roi_params, 'lr': lr * roi_lr_alpha, 'name': 'roi_heads'},
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=0.01)
        self.scheduler = None
        
        print(f"Optimizer parameter groups:")
        print(f"  Base params: {len(base_params)} tensors, lr={lr:.2e}")
        print(f"  RoI params:  {len(roi_params)} tensors, lr={lr * roi_lr_alpha:.2e} (alpha={roi_lr_alpha})")

        # AMP scaler for training only (not used in LR finder)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Scheduler type (set in _create_scheduler)
        self._scheduler_type = "plateau"

        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

        # Initialize W&B if project is specified
        if wandb_project is not None:
            if not WANDB_AVAILABLE:
                raise ImportError(
                    "wandb is required for W&B logging. Install with: pip install wandb"
                )

            # Build hyperparameters dict if not provided
            if hyperparameters is None:
                hyperparameters = {}

            # Add trainer config to hyperparameters
            hyperparameters.update(
                {
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "num_classes": num_classes,
                    "use_amp": use_amp,
                    "device": str(self.device),
                    "train_samples": len(self.train_dataset),
                    "val_samples": len(self.val_dataset),
                }
            )

            # Create W&B config
            wandb_config = WandbConfig(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
                tags=wandb_tags or [],
                notes=wandb_notes,
                log_freq=wandb_log_freq,
                num_val_images=wandb_num_val_images,
                conf_threshold=wandb_conf_threshold,
            )

            # Initialize logger
            self.wandb_logger = WandbLogger(wandb_config, hyperparameters)

            # Setup fixed validation images for consistent visualization
            self._setup_validation_images(wandb_num_val_images)

            print(f"W&B logging enabled: {self.wandb_logger.run.url}")

    def _setup_validation_images(self, num_images: int = 4):
        """Setup fixed validation images for consistent W&B visualization."""
        # Select evenly spaced indices from validation set
        val_len = len(self.val_dataset)
        step = max(1, val_len // num_images)
        self._val_image_indices = [
            min(i * step, val_len - 1) for i in range(num_images)
        ]

        if self.wandb_logger is not None:
            self.wandb_logger.set_validation_images(self._val_image_indices)

        # Pre-load validation images
        self._fixed_val_images, self._fixed_val_targets = get_fixed_val_batch(
            self.val_dataset, self._val_image_indices, str(self.device)
        )
        print(
            f"Selected {len(self._val_image_indices)} validation images for visualization"
        )

    def _create_scheduler(self):
        """Create ReduceLROnPlateau learning rate scheduler monitoring val_mAP."""
        # ReduceLROnPlateau: reduces LR when validation mAP plateaus
        # - mode='max': reduce LR when metric stops increasing (mAP should increase)
        # - factor=0.5: halve the LR on plateau
        # - patience=3: wait 3 epochs before reducing (mAP is noisier than loss)
        # - threshold=1e-3: minimum change to qualify as improvement
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            threshold=1e-3,
        )
        print("Using ReduceLROnPlateau scheduler (steps on validation mAP)")

    def find_lr(
        self,
        start_lr: float = 1e-6,
        end_lr: float = 1e-2,
        num_iter: int = 100,
        smooth_factor: float = 0.05,
        plot: bool = True,
    ) -> float:
        """
        Learning Rate Finder using the LR range test.

        This is a DIAGNOSTIC TOOL ONLY - it does NOT update model weights.
        It performs forward + backward passes to measure loss response at different LRs,
        then restores the exact initial state.

        Args:
            start_lr: Starting learning rate (default 1e-6, safe for detection)
            end_lr: Maximum learning rate to test (default 1e-2, avoids explosion)
            num_iter: Number of iterations
            smooth_factor: Smoothing factor for loss curve (exponential moving average)
            plot: Whether to plot the results

        Returns:
            Suggested learning rate (LR at minimum smoothed loss / 3)
        """
        print("=" * 60)
        print("Learning Rate Finder (diagnostic only - no weight updates)")
        print("=" * 60)

        # Save exact initial state for restoration
        # Deep copy all tensors to ensure we can restore exactly
        initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        initial_optimizer_state = self.optimizer.state_dict()
        initial_lr = self.lr

        # Setup - NO AMP for deterministic results
        self.model.train()
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr

        lrs = []
        losses = []
        smoothed_loss = 0
        best_loss = float("inf")
        best_lr = start_lr

        # Create iterator
        data_iter = iter(self.train_loader)

        pbar = tqdm(range(num_iter), desc="Finding LR")
        for iteration in pbar:
            # Get batch (cycle if needed)
            try:
                images, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                images, targets = next(data_iter)

            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Skip empty batches
            if all(len(t["boxes"]) == 0 for t in targets):
                continue

            # Zero gradients before forward pass
            self.optimizer.zero_grad()

            try:
                # Forward pass - NO AMP for consistent, deterministic results
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                # Check for invalid loss
                if not torch.isfinite(loss):
                    print(f"\nNaN/Inf loss at lr={lr:.2e}, stopping")
                    break

                # Backward pass ONLY - to measure gradient response
                # NO optimizer.step() - we don't want to update weights
                loss.backward()

                loss_val = loss.item()

                # Exponential moving average smoothing
                if iteration == 0:
                    smoothed_loss = loss_val
                else:
                    smoothed_loss = (
                        smooth_factor * loss_val + (1 - smooth_factor) * smoothed_loss
                    )

                # Track best loss and corresponding LR
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss
                    best_lr = lr

                # Early stopping if loss diverges (4x best loss)
                if smoothed_loss > 4 * best_loss and iteration > 10:
                    print(f"\nStopping early - loss exploding at lr={lr:.2e}")
                    break

                lrs.append(lr)
                losses.append(smoothed_loss)

                pbar.set_postfix(
                    lr=f"{lr:.2e}", loss=f"{smoothed_loss:.4f}", best=f"{best_loss:.4f}"
                )

            except RuntimeError as e:
                print(f"\nError at lr={lr:.2e}: {e}")
                break

            # Increase LR for next iteration (exponential schedule)
            lr *= lr_mult

            # Clean up tensors
            del images, targets, loss, loss_dict

        # Restore EXACT initial state - model never actually changed due to no optimizer.step()
        # but we restore anyway for safety and to reset any gradient accumulation
        self.model.load_state_dict(initial_model_state)
        self.optimizer.load_state_dict(initial_optimizer_state)
        self.lr = initial_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = initial_lr

        # Determine suggested LR
        if len(losses) < 10:
            print("Not enough data points collected. Using default LR.")
            return self.lr

        # Suggested LR = LR at minimum smoothed loss / 3
        # Dividing by 3 provides a safety margin for stable training
        suggested_lr = best_lr / 3

        print(f"\n{'=' * 60}")
        print(f"Best smoothed loss: {best_loss:.4f} at LR: {best_lr:.2e}")
        print(f"Suggested Learning Rate: {suggested_lr:.6f} (best_lr / 3)")
        print(f"{'=' * 60}")

        if plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(lrs, losses, "b-", linewidth=2, label="Smoothed Loss")
                ax.axvline(
                    x=best_lr,
                    color="orange",
                    linestyle=":",
                    label=f"Min loss LR: {best_lr:.2e}",
                )
                ax.axvline(
                    x=suggested_lr,
                    color="r",
                    linestyle="--",
                    label=f"Suggested LR: {suggested_lr:.2e}",
                )
                ax.set_xscale("log")
                ax.set_xlabel("Learning Rate (log scale)")
                ax.set_ylabel("Loss (smoothed)")
                ax.set_title("Learning Rate Finder")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Matplotlib not available for plotting")

        gc.collect()
        torch.cuda.empty_cache()

        return suggested_lr

    def set_lr(self, lr: float):
        """Set learning rate for optimizer."""
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Learning rate set to: {lr:.6f}")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def compute_map(
        self,
        dataloader: DataLoader,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        max_samples: int = 200,
        max_dets_per_image: int = 100,
        max_preds_per_class: int = 2000,
    ) -> Tuple[float, float]:
        """
        Compute mAP@IoU and mean IoU on a dataset (memory-efficient version).

        Args:
            dataloader: DataLoader to evaluate on
            iou_threshold: IoU threshold for mAP computation
            score_threshold: Minimum score for predictions
            max_samples: Maximum samples to evaluate (default 200 for speed)
            max_dets_per_image: Max detections per image
            max_preds_per_class: Max predictions to keep per class (prunes low scores)

        Returns:
            Tuple of (mAP@IoU, mean_IoU)
        """
        self.model.eval()

        # Collect predictions and ground truths per class
        # Store as single arrays that we extend, not lists of arrays
        class_data: Dict[int, Dict[str, List]] = {}
        iou_sum = 0.0
        iou_count = 0

        num_samples = 0
        for images, targets in dataloader:
            if max_samples and num_samples >= max_samples:
                break

            images = [img.to(self.device) for img in images]
            predictions = self.model(images)

            for pred, target in zip(predictions, targets):
                # Filter by score and limit detections
                scores = pred["scores"]
                keep = scores >= score_threshold
                if keep.sum() > max_dets_per_image:
                    _, top_idx = scores.topk(max_dets_per_image)
                    keep = torch.zeros_like(keep, dtype=torch.bool)
                    keep[top_idx] = True

                # Move to CPU as numpy
                pred_boxes = pred["boxes"][keep].cpu().numpy().astype(np.float32)
                pred_labels = pred["labels"][keep].cpu().numpy()
                pred_scores = pred["scores"][keep].cpu().numpy().astype(np.float32)

                gt_boxes = target["boxes"].cpu().numpy().astype(np.float32)
                gt_labels = target["labels"].cpu().numpy()

                # Compute mean IoU incrementally (avoid storing all IoUs)
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = self._compute_iou_numpy(pred_boxes, gt_boxes)
                    best_ious = ious.max(axis=0)
                    iou_sum += best_ious.sum()
                    iou_count += len(best_ious)
                    del ious, best_ious

                # Collect per-class data
                all_labels = set(pred_labels.tolist()) | set(gt_labels.tolist())
                for label_id in all_labels:
                    if label_id not in class_data:
                        class_data[label_id] = {
                            "pred_boxes": [],
                            "pred_scores": [],
                            "gt_boxes": [],
                            "n_preds": 0,
                            "n_gts": 0,
                        }

                    pred_mask = pred_labels == label_id
                    if pred_mask.any():
                        boxes = pred_boxes[pred_mask]
                        scores_cls = pred_scores[pred_mask]

                        # Only add if under limit, or if scores are high enough
                        data = class_data[label_id]
                        if data["n_preds"] < max_preds_per_class:
                            data["pred_boxes"].append(boxes)
                            data["pred_scores"].append(scores_cls)
                            data["n_preds"] += len(boxes)

                    gt_mask = gt_labels == label_id
                    if gt_mask.any():
                        class_data[label_id]["gt_boxes"].append(gt_boxes[gt_mask])
                        class_data[label_id]["n_gts"] += gt_mask.sum()

            num_samples += len(images)

            # Clear references
            del images, predictions

        # Compute AP per class (one at a time to limit memory)
        aps = []
        for label_id in list(class_data.keys()):
            data = class_data[label_id]

            if data["pred_boxes"]:
                all_pred_boxes = np.concatenate(data["pred_boxes"])
                all_pred_scores = np.concatenate(data["pred_scores"])
            else:
                all_pred_boxes = np.zeros((0, 4), dtype=np.float32)
                all_pred_scores = np.zeros(0, dtype=np.float32)

            if data["gt_boxes"]:
                all_gt_boxes = np.concatenate(data["gt_boxes"])
            else:
                all_gt_boxes = np.zeros((0, 4), dtype=np.float32)

            ap = self._compute_ap_numpy(
                all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold
            )
            aps.append(ap)

            # Free memory immediately after each class
            del all_pred_boxes, all_pred_scores, all_gt_boxes
            del class_data[label_id]

        mAP = sum(aps) / len(aps) if aps else 0.0
        mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

        self.model.train()
        return mAP, mean_iou

    def _compute_iou_numpy(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Fast IoU computation using numpy."""
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        return inter / (union + 1e-6)

    def _compute_ap_numpy(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        iou_threshold: float,
    ) -> float:
        """Fast AP computation using numpy."""
        if len(gt_boxes) == 0:
            return 1.0 if len(pred_boxes) == 0 else 0.0
        if len(pred_boxes) == 0:
            return 0.0

        # Sort by score
        sorted_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_idx]

        # Compute IoU
        ious = self._compute_iou_numpy(pred_boxes, gt_boxes)

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        for i in range(len(pred_boxes)):
            iou_max = ious[i].max()
            gt_idx = ious[i].argmax()

            if iou_max >= iou_threshold and not gt_matched[gt_idx]:
                tp[i] = 1
                gt_matched[gt_idx] = True
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.any():
                ap += precisions[mask].max() / 11

        return ap

    def _compute_gradient_norm(self) -> float:
        """
        Compute global L2 gradient norm across all parameters.

        This metric helps diagnose training stability - high or fluctuating
        gradient norms may indicate learning rate issues.
        """
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm_sq)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loss_accumulator = {}

        # Training dynamics tracking
        batch_losses = []  # For loss variance computation
        gradient_norms = []  # For epoch-averaged gradient norm

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")

        for images, targets in pbar:
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            self.optimizer.zero_grad(set_to_none=True)

            # Skip batches with no valid targets
            valid_targets = [t for t in targets if len(t["boxes"]) > 0]
            if len(valid_targets) == 0:
                continue

            try:
                if self.use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        loss_dict = self.model(images, targets)
                        loss = sum(loss_dict.values())

                    # Check for NaN/Inf before backward
                    if not torch.isfinite(loss):
                        print(f"Warning: NaN/Inf loss detected, skipping batch")
                        self.optimizer.zero_grad(set_to_none=True)
                        del images, targets, loss, loss_dict
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients (GradScaler will skip step if NaN/Inf detected)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())

                    # Check for NaN/Inf before backward
                    if not torch.isfinite(loss):
                        print(f"Warning: NaN/Inf loss detected, skipping batch")
                        del images, targets, loss, loss_dict
                        torch.cuda.empty_cache()
                        continue

                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value
                batch_losses.append(loss_value)  # Track for variance computation

                # Compute gradient norm AFTER backward, BEFORE optimizer step
                # This measures the actual gradient magnitude used for updates
                grad_norm = self._compute_gradient_norm()
                gradient_norms.append(grad_norm)

                # Accumulate individual losses
                for k, v in loss_dict.items():
                    loss_accumulator.setdefault(k, 0.0)
                    loss_accumulator[k] += v.item()

                # =========================================================
                # W&B LOGGING: Training step metrics
                # =========================================================
                if self.wandb_logger is not None:
                    current_lr = self.get_lr()

                    # Log training metrics
                    self.wandb_logger.log_training_step(
                        loss_dict=loss_dict,
                        learning_rate=current_lr,
                        step=self._global_step,
                        epoch=epoch,
                    )

                    # Log gradient norms for CBAM and RoI layers
                    self.wandb_logger.log_gradient_norms(
                        self.model, step=self._global_step
                    )

                self._global_step += 1

                # Show all losses in pbar with current LR
                loss_postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                loss_postfix["total"] = f"{loss_value:.4f}"
                loss_postfix["lr"] = f"{self.get_lr():.2e}"
                pbar.set_postfix(loss=loss_postfix)

            except RuntimeError as e:
                print(f"Warning: RuntimeError in batch: {e}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            del images, targets, loss, loss_dict

        # Note: ReduceLROnPlateau scheduler steps on validation loss in fit()

        # Compute training dynamics metrics
        # Loss variance: measures stability of training within epoch
        # High variance may indicate noisy gradients or LR issues
        if len(batch_losses) > 1:
            mean_loss = sum(batch_losses) / len(batch_losses)
            loss_variance = sum((l - mean_loss) ** 2 for l in batch_losses) / (
                len(batch_losses) - 1
            )
        else:
            loss_variance = 0.0

        # Epoch-averaged gradient norm
        avg_grad_norm = (
            sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        )

        # Return average losses plus training dynamics
        avg_losses = {
            k: v / len(self.train_loader) for k, v in loss_accumulator.items()
        }
        avg_losses["total"] = total_loss / len(self.train_loader)
        avg_losses["grad_norm"] = avg_grad_norm
        avg_losses["loss_variance"] = loss_variance
        return avg_losses

    @torch.no_grad()
    def validate(self):
        self.model.train()
        total_loss = 0.0
        loss_accumulator = {}

        pbar = tqdm(self.val_loader, desc="Validation")

        for images, targets in pbar:
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            with (
                autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else torch.no_grad()
            ):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict[k] for k in loss_dict)
            total_loss += loss.item()
            # Accumulate individual losses
            for k, v in loss_dict.items():
                loss_accumulator.setdefault(k, 0.0)
                loss_accumulator[k] += v.item()
            # Show all losses in pbar
            loss_postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
            loss_postfix["total"] = f"{loss.item():.4f}"
            pbar.set_postfix(loss=loss_postfix)

            del images, targets, loss, loss_dict

        # Return average losses
        avg_losses = {k: v / len(self.val_loader) for k, v in loss_accumulator.items()}
        avg_losses["total"] = total_loss / len(self.val_loader)
        return avg_losses

    def save_checkpoint(self, path, epoch, val_loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

    def load_checkpoint(self, path):
        """Load model from checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        return checkpoint

    def fit(
        self,
        epochs=20,
        save_dir="checkpoints",
        find_lr_first=False,
        compute_metrics_every: int = 1,
        max_map_samples: int = None,
        # Early stopping based on mAP gap
        early_stop_gap_patience: int = None,
        early_stop_gap_threshold: float = 0.01,
    ) -> Dict[str, List[float]]:
        """
        Train the model with optional W&B logging and early stopping.

        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            find_lr_first: If True, run LR finder before training
            compute_metrics_every: Compute mAP metrics every N epochs (default=1)
            max_map_samples: Max samples for mAP computation (None=all, use smaller for speed)
            early_stop_gap_patience: Stop training if the train-val mAP gap doesn't improve
                for this many epochs. Set to None to disable. Default: None
            early_stop_gap_threshold: Minimum improvement in mAP gap to reset patience counter.
                Default: 0.01 (1% improvement)

        Returns:
            history: Dictionary of metric lists, compatible with TensorBoard logging.
                Keys follow convention: 'train/<metric>', 'val/<metric>', 'train_val/<metric>'

        Early Stopping Logic (mAP gap):
            The train-val mAP gap indicates overfitting when positive (train >> val).
            We stop training if the gap doesn't decrease (improve) for `early_stop_gap_patience` epochs.
            This catches cases where the model keeps overfitting without validation improvement.
        """
        # Initialize history dictionary for all metrics
        # TensorBoard-compatible naming: prefix/metric_name
        history: Dict[str, List[float]] = {
            # Loss metrics (existing)
            "train/loss": [],
            "val/loss": [],
            # Performance metrics (mAP, IoU)
            "train/mAP@0.5": [],
            "val/mAP@0.5": [],
            "val/mean_iou": [],
            # Training dynamics metrics
            "train/grad_norm": [],
            "train/loss_variance": [],
            "train_val/mAP_gap": [],
            # Learning rate tracking
            "train/lr": [],
        }

        # Optionally find best LR before training
        if find_lr_first:
            suggested_lr = self.find_lr()
            self.set_lr(suggested_lr)

        # Create scheduler
        self._create_scheduler()

        best_loss = float("inf")
        best_map = 0.0
        best_train_map = 0.0

        # Early stopping state (mAP gap based)
        # We want the ABSOLUTE gap to DECREASE (closer to 0 = better alignment)
        # Gap can be positive (overfitting) or negative (underfitting)
        best_gap_abs = float("inf")
        gap_patience_counter = 0

        # Update W&B logger epoch tracking
        if self.wandb_logger is not None:
            self.wandb_logger.epoch = 0

        for epoch in range(1, epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{epochs} | LR: {self.get_lr():.2e}")
            print(f"{'=' * 60}")

            # Update W&B logger epoch
            if self.wandb_logger is not None:
                self.wandb_logger.epoch = epoch

            start = time.time()

            # Training epoch (includes gradient norm and loss variance)
            train_losses = self.train_epoch(epoch)

            # Validation (loss only)
            val_losses = self.validate()

            # Compute performance metrics (mAP, IoU)
            if epoch % compute_metrics_every == 0:
                print("Computing mAP metrics...")
                # Validation mAP and mean IoU (primary metrics)
                val_map, val_mean_iou = self.compute_map(
                    self.val_loader,
                    iou_threshold=0.5,
                    max_samples=max_map_samples if max_map_samples else 200,
                )
                # Training mAP (for overfitting diagnosis)
                train_map, _ = self.compute_map(
                    self.train_loader,
                    iou_threshold=0.5,
                    max_samples=max_map_samples if max_map_samples else 200,
                )
                # mAP gap: positive = overfitting, negative = underfitting
                map_gap = train_map - val_map
            else:
                # Use previous values if not computing this epoch
                val_map = history["val/mAP@0.5"][-1] if history["val/mAP@0.5"] else 0.0
                val_mean_iou = (
                    history["val/mean_iou"][-1] if history["val/mean_iou"] else 0.0
                )
                train_map = (
                    history["train/mAP@0.5"][-1] if history["train/mAP@0.5"] else 0.0
                )
                map_gap = (
                    history["train_val/mAP_gap"][-1]
                    if history["train_val/mAP_gap"]
                    else 0.0
                )

            epoch_time = time.time() - start

            # Record metrics in history
            history["train/loss"].append(train_losses["total"])
            history["val/loss"].append(val_losses["total"])
            history["train/mAP@0.5"].append(train_map)
            history["val/mAP@0.5"].append(val_map)
            history["val/mean_iou"].append(val_mean_iou)
            history["train/grad_norm"].append(train_losses["grad_norm"])
            history["train/loss_variance"].append(train_losses["loss_variance"])
            history["train_val/mAP_gap"].append(map_gap)
            history["train/lr"].append(self.get_lr())

            # =========================================================
            # W&B LOGGING: Epoch-level validation metrics
            # =========================================================
            if self.wandb_logger is not None:
                # Log validation loss and metrics
                val_metrics = {
                    "mAP@0.5": val_map,
                    "mean_iou": val_mean_iou,
                    "train_mAP@0.5": train_map,
                    "mAP_gap": map_gap,
                }
                self.wandb_logger.log_validation_metrics(
                    val_loss=val_losses["total"],
                    val_metrics=val_metrics,
                    epoch=epoch,
                )

                # Log validation predictions visualization
                self._log_validation_predictions(epoch)

            # Print detailed results
            print(f"\nEpoch {epoch} Results (Time: {epoch_time:.1f}s):")
            print(f"  Losses:")
            print(f"    Train: {train_losses['total']:.4f}")
            print(f"    Val:   {val_losses['total']:.4f}")
            print(f"  Performance Metrics:")
            print(f"    Train mAP@0.5: {train_map:.4f}")
            print(f"    Val mAP@0.5:   {val_map:.4f} (primary metric)")
            print(f"    Val Mean IoU:  {val_mean_iou:.4f}")
            print(f"  Training Dynamics:")
            print(f"    Gradient Norm: {train_losses['grad_norm']:.4f}")
            print(f"    Loss Variance: {train_losses['loss_variance']:.6f}")
            print(f"    mAP Gap (train-val): {map_gap:+.4f}")
            print(f"  Detailed Train Losses:")
            for k, v in train_losses.items():
                if k not in ["total", "grad_norm", "loss_variance"]:
                    print(f"    {k}: {v:.4f}")

            val_loss = val_losses["total"]

            # Step ReduceLROnPlateau scheduler with validation mAP
            if self.scheduler is not None:
                self.scheduler.step(val_map)

            self.save_checkpoint(f"{save_dir}/last.pth", epoch, val_loss)

            # Save best model based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = f"{save_dir}/best.pth"
                self.save_checkpoint(best_model_path, epoch, val_loss)
                print("-> New best model saved (by loss)")

                # Log best model as W&B artifact
                if self.wandb_logger is not None:
                    self.wandb_logger.log_best_model(best_model_path, val_loss, val_map)

            # Also track best mAP
            if val_map > best_map:
                best_map = val_map
                self.save_checkpoint(f"{save_dir}/best_map.pth", epoch, val_loss)
                print(f"-> New best val mAP@0.5: {best_map:.4f}")

            # Track best training mAP (separate artifact for overfitting analysis)
            if train_map > best_train_map:
                best_train_map = train_map
                best_train_map_path = f"{save_dir}/best_train_map.pth"
                self.save_checkpoint(best_train_map_path, epoch, val_loss)
                print(f"-> New best train mAP@0.5: {best_train_map:.4f}")

                # Log to W&B as separate artifact
                if self.wandb_logger is not None:
                    self.wandb_logger.log_best_train_map_model(
                        best_train_map_path, train_map, val_map
                    )

            # =========================================================
            # EARLY STOPPING: Based on train-val mAP gap (DISABLED BY DEFAULT)
            # =========================================================
            # The absolute gap should ideally decrease (train and val closer together).
            # Gap can be positive (train > val = overfitting) or negative (val > train).
            # We want |gap| to decrease, meaning better alignment between train and val.
            if (
                early_stop_gap_patience is not None
                and epoch % compute_metrics_every == 0
            ):
                gap_abs = abs(map_gap)
                # Check if absolute gap improved (decreased) by at least threshold
                if gap_abs < best_gap_abs - early_stop_gap_threshold:
                    best_gap_abs = gap_abs
                    gap_patience_counter = 0
                    print(f"-> mAP gap improved: |{map_gap:+.4f}| = {gap_abs:.4f}")
                else:
                    gap_patience_counter += 1
                    print(
                        f"-> mAP gap patience: {gap_patience_counter}/{early_stop_gap_patience} (|gap|={gap_abs:.4f})"
                    )

                # Check if we should stop
                if gap_patience_counter >= early_stop_gap_patience:
                    print(f"\n{'=' * 60}")
                    print(
                        f"EARLY STOPPING: mAP gap not improving for {early_stop_gap_patience} epochs"
                    )
                    print(f"  Current gap: {map_gap:+.4f} (|gap|={gap_abs:.4f})")
                    print(f"  Best |gap|: {best_gap_abs:.4f}")
                    print(
                        f"  Train and val performance are not converging - consider stopping."
                    )
                    print(f"{'=' * 60}")
                    break

            gc.collect()
            torch.cuda.empty_cache()

        # Determine if training completed normally or was stopped early
        stopped_early = (
            early_stop_gap_patience is not None
            and gap_patience_counter >= early_stop_gap_patience
        )

        print(f"\n{'=' * 60}")
        if stopped_early:
            print(f"Training stopped early at epoch {epoch}/{epochs}")
        else:
            print(f"Training complete!")
        print(f"  Best val loss: {best_loss:.4f}")
        print(f"  Best val mAP@0.5: {best_map:.4f}")
        if early_stop_gap_patience is not None:
            print(f"  Best |mAP gap|: {best_gap_abs:.4f}")
        print(f"{'=' * 60}")

        return history

    @torch.no_grad()
    def _log_validation_predictions(self, epoch: int):
        """Log validation predictions to W&B."""
        if self.wandb_logger is None or self._fixed_val_images is None:
            return

        self.model.eval()

        # Get predictions on fixed validation images
        val_images_device = [img.to(self.device) for img in self._fixed_val_images]
        predictions = self.model(val_images_device)

        # Move predictions back to CPU for logging
        predictions_cpu = [
            {k: v.cpu() for k, v in pred.items()} for pred in predictions
        ]

        # Log visualization
        self.wandb_logger.log_validation_predictions(
            images=self._fixed_val_images,
            targets=self._fixed_val_targets,
            predictions=predictions_cpu,
            epoch=epoch,
        )

        self.model.train()

    def finish(self):
        """Finish the W&B run and clean up resources."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish()
            print("W&B run finished.")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures W&B run is finished."""
        self.finish()
        return False

    @torch.no_grad()
    def visualize_predictions(
        self, num_samples=5, score_threshold=0.5, save_path=None, mask_alpha=0.4
    ):
        """Visualize model predictions on validation set with masks.

        Args:
            num_samples: Number of validation samples to visualize
            score_threshold: Minimum confidence score for predictions
            save_path: Optional path to save the figure
            mask_alpha: Transparency for mask overlay (0-1)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        self.model.eval()

        # Colors for different instances (distinct colors)
        COLORS = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0],
            [1, 0, 0.5],
            [0.5, 1, 0],
            [0, 1, 0.5],
        ]

        # Get category names if available
        if hasattr(self.val_dataset, "dataset"):
            dataset = self.val_dataset.dataset
        else:
            dataset = self.val_dataset

        category_names = (
            dataset.get_category_names()
            if hasattr(dataset, "get_category_names")
            else {}
        )

        # Select random samples
        indices = np.random.choice(
            len(self.val_dataset),
            min(num_samples, len(self.val_dataset)),
            replace=False,
        )

        fig, axes = plt.subplots(num_samples, 2, figsize=(16, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for idx, sample_idx in enumerate(indices):
            image, target = self.val_dataset[sample_idx]

            # Convert image for visualization and denormalize if needed
            if isinstance(image, torch.Tensor):
                img_display = image.permute(1, 2, 0).cpu().numpy()
            else:
                img_display = np.array(image)

            # Check if image is normalized (values outside 0-1 range) and denormalize
            if img_display.min() < 0 or img_display.max() > 1:
                # Denormalize using ImageNet mean/std
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = img_display * std + mean

            # Clip to valid range and convert to displayable format
            img_display = np.clip(img_display, 0, 1)

            # Get predictions
            # Ensure image is a tensor and move to device
            if not isinstance(image, torch.Tensor):
                import torchvision.transforms.functional as F

                image = F.to_tensor(image)
            image_tensor = image.to(self.device)
            predictions = self.model([image_tensor])[0]

            # Filter predictions by score
            keep_idx = predictions["scores"] > score_threshold
            pred_boxes = predictions["boxes"][keep_idx].cpu().numpy()
            pred_labels = predictions["labels"][keep_idx].cpu().numpy()
            pred_scores = predictions["scores"][keep_idx].cpu().numpy()
            pred_masks = (
                predictions["masks"][keep_idx].cpu().numpy()
                if "masks" in predictions
                else None
            )

            # Ground truth visualization with masks
            ax_gt = axes[idx, 0]
            img_gt_overlay = img_display.copy()

            # Overlay ground truth masks
            if "masks" in target and len(target["masks"]) > 0:
                from scipy.ndimage import binary_dilation, binary_erosion

                gt_masks = target["masks"].cpu().numpy()
                gt_labels_np = target["labels"].cpu().numpy()
                for i, (mask, label) in enumerate(zip(gt_masks, gt_labels_np)):
                    color = np.array(COLORS[int(label) % len(COLORS)])
                    # Handle different mask formats (H,W) or (1,H,W)
                    if mask.ndim == 3:
                        mask = mask[0]
                    mask_bool = mask > 0.5
                    # Fill mask with color
                    img_gt_overlay[mask_bool] = (
                        img_gt_overlay[mask_bool] * (1 - mask_alpha)
                        + color * mask_alpha
                    )
                    # Add dark contour outline for visibility
                    dilated = binary_dilation(mask_bool, iterations=2)
                    eroded = binary_erosion(mask_bool, iterations=1)
                    contour = dilated & ~eroded
                    img_gt_overlay[contour] = color * 0.3  # Darker outline

            ax_gt.imshow(img_gt_overlay)
            ax_gt.set_title(f"Ground Truth (Sample {sample_idx})")
            ax_gt.axis("off")

            if "boxes" in target and len(target["boxes"]) > 0:
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels_np = target["labels"].cpu().numpy()

                for i, (box, label) in enumerate(zip(gt_boxes, gt_labels_np)):
                    x1, y1, x2, y2 = box
                    color = COLORS[int(label) % len(COLORS)]
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax_gt.add_patch(rect)

                    cat_name = category_names.get(int(label), f"Class {label}")
                    ax_gt.text(
                        x1,
                        y1 - 5,
                        cat_name,
                        color="white",
                        fontsize=8,
                        bbox=dict(facecolor=color, alpha=0.7),
                    )

            # Predictions visualization with masks
            ax_pred = axes[idx, 1]
            img_pred_overlay = img_display.copy()
            img_h, img_w = img_pred_overlay.shape[:2]

            # Overlay predicted masks
            if pred_masks is not None and len(pred_masks) > 0:
                from scipy.ndimage import zoom, binary_dilation, binary_erosion

                for i, (mask, box, label) in enumerate(
                    zip(pred_masks, pred_boxes, pred_labels)
                ):
                    color = np.array(COLORS[int(label) % len(COLORS)])
                    # Predicted masks are (1, H, W) with probabilities
                    if mask.ndim == 3:
                        mask = mask[0]

                    # Check if mask needs to be resized (raw 28x28 mask head output)
                    mask_h, mask_w = mask.shape
                    if mask_h != img_h or mask_w != img_w:
                        # Resize mask to bounding box size and paste onto full image
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_w, x2), min(img_h, y2)
                        box_h, box_w = y2 - y1, x2 - x1

                        if box_h > 0 and box_w > 0:
                            # Resize mask to box dimensions
                            scale_h, scale_w = box_h / mask_h, box_w / mask_w
                            resized_mask = zoom(mask, (scale_h, scale_w), order=1)

                            # Create full-size mask and paste
                            full_mask = np.zeros((img_h, img_w), dtype=np.float32)
                            # Handle potential size mismatches due to rounding
                            paste_h = min(resized_mask.shape[0], img_h - y1)
                            paste_w = min(resized_mask.shape[1], img_w - x1)
                            full_mask[y1 : y1 + paste_h, x1 : x1 + paste_w] = (
                                resized_mask[:paste_h, :paste_w]
                            )
                            mask = full_mask

                    mask_bool = mask > 0.5
                    # Fill mask with color
                    img_pred_overlay[mask_bool] = (
                        img_pred_overlay[mask_bool] * (1 - mask_alpha)
                        + color * mask_alpha
                    )
                    # Add dark contour outline for visibility
                    dilated = binary_dilation(mask_bool, iterations=2)
                    eroded = binary_erosion(mask_bool, iterations=1)
                    contour = dilated & ~eroded
                    img_pred_overlay[contour] = color * 0.3  # Darker outline

            ax_pred.imshow(img_pred_overlay)
            ax_pred.set_title(
                f"Predictions (Sample {sample_idx}, {len(pred_boxes)} detections)"
            )
            ax_pred.axis("off")

            for i, (box, label, score) in enumerate(
                zip(pred_boxes, pred_labels, pred_scores)
            ):
                x1, y1, x2, y2 = box
                color = COLORS[int(label) % len(COLORS)]
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax_pred.add_patch(rect)

                cat_name = category_names.get(int(label), f"Class {label}")
                ax_pred.text(
                    x1,
                    y1 - 5,
                    f"{cat_name} {score:.2f}",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor=color, alpha=0.7),
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        plt.show()

        self.model.train()


def train(
    data_root,
    epochs=20,
    batch_size=1,
    lr=0.005,
    device="cuda",
):
    trainer = Trainer(
        data_root=data_root,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    trainer.fit(epochs=epochs)
    return trainer
