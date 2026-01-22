"""
Weights & Biases (W&B) integration for iSAID instance segmentation training.

This module provides comprehensive logging capabilities including:
- Training metrics and loss components
- Gradient norms for custom layers (CBAM, RoI heads)
- Interactive visualization of predictions vs ground truth
- Model checkpointing as W&B artifacts
"""

import os
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# =============================================================================
# iSAID Class Labels (0-15)
# =============================================================================
ISAID_CLASS_LABELS = {
    0: "background",
    1: "ship",
    2: "storage_tank",
    3: "baseball_diamond",
    4: "tennis_court",
    5: "basketball_court",
    6: "ground_track_field",
    7: "bridge",
    8: "large_vehicle",
    9: "small_vehicle",
    10: "helicopter",
    11: "swimming_pool",
    12: "roundabout",
    13: "soccer_ball_field",
    14: "plane",
    15: "harbor",
}

# Color palette for visualization (RGB)
ISAID_CLASS_COLORS = {
    0: [0, 0, 0],        # background - black
    1: [255, 0, 0],      # ship - red
    2: [0, 255, 0],      # storage_tank - green
    3: [0, 0, 255],      # baseball_diamond - blue
    4: [255, 255, 0],    # tennis_court - yellow
    5: [255, 0, 255],    # basketball_court - magenta
    6: [0, 255, 255],    # ground_track_field - cyan
    7: [128, 0, 0],      # bridge - maroon
    8: [0, 128, 0],      # large_vehicle - dark green
    9: [0, 0, 128],      # small_vehicle - navy
    10: [128, 128, 0],   # helicopter - olive
    11: [128, 0, 128],   # swimming_pool - purple
    12: [0, 128, 128],   # roundabout - teal
    13: [255, 128, 0],   # soccer_ball_field - orange
    14: [128, 255, 0],   # plane - lime
    15: [255, 0, 128],   # harbor - pink
}


@dataclass
class WandbConfig:
    """Configuration for W&B logging."""
    project: str = "isaid-custom-segmentation"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    log_freq: int = 20  # Log every N batches
    log_gradients: bool = True
    log_images: bool = True
    num_val_images: int = 4  # Number of validation images to visualize
    conf_threshold: float = 0.5  # Confidence threshold for predictions


class WandbLogger:
    """
    Weights & Biases logger for Mask R-CNN training.
    
    Handles:
    - Initialization with hyperparameters
    - Training loss logging
    - Gradient norm logging for custom layers
    - Validation image visualization
    - Model checkpointing as artifacts
    """
    
    def __init__(
        self,
        config: WandbConfig,
        hyperparameters: Dict[str, Any],
        resume: bool = False,
        run_id: Optional[str] = None,
    ):
        """
        Initialize W&B run.
        
        Args:
            config: WandbConfig with project settings
            hyperparameters: Dictionary of all training hyperparameters
            resume: Whether to resume a previous run
            run_id: Run ID to resume (if resume=True)
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required for WandbLogger. "
                "Install with: pip install wandb"
            )
        
        self.config = config
        self.hyperparameters = hyperparameters
        self._step = 0
        self._epoch = 0
        self._val_image_indices = None
        self._initialized = False
        
        # Initialize wandb
        self.run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.run_name,
            tags=config.tags,
            notes=config.notes,
            config=hyperparameters,
            resume="must" if resume else "allow",
            id=run_id,
        )
        
        # Store class labels in config
        wandb.config.update({"class_labels": ISAID_CLASS_LABELS}, allow_val_change=True)
        
        self._initialized = True
        print(f"W&B run initialized: {self.run.name}")
        print(f"View at: {self.run.url}")
    
    @property
    def step(self) -> int:
        return self._step
    
    @step.setter
    def step(self, value: int):
        self._step = value
    
    @property
    def epoch(self) -> int:
        return self._epoch
    
    @epoch.setter 
    def epoch(self, value: int):
        self._epoch = value
    
    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step % self.config.log_freq == 0
    
    # =========================================================================
    # Training Loop Logging (Scalars)
    # =========================================================================
    
    def log_training_step(
        self,
        loss_dict: Dict[str, torch.Tensor],
        learning_rate: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """
        Log training metrics for a single step.
        
        Args:
            loss_dict: Dictionary of loss components from model
            learning_rate: Current learning rate
            step: Global step number (optional, uses internal counter)
            epoch: Current epoch (optional)
        """
        if step is not None:
            self._step = step
        if epoch is not None:
            self._epoch = epoch
            
        if not self.should_log(self._step):
            return
        
        # Extract loss values
        metrics = {
            "train/learning_rate": learning_rate,
            "train/epoch": self._epoch,
            "train/step": self._step,
        }
        
        # Map loss components to wandb metric names
        loss_mapping = {
            "loss_classifier": "train/roi_class_loss",
            "loss_box_reg": "train/roi_box_loss",
            "loss_mask": "train/mask_loss",
            "loss_objectness": "train/rpn_class_loss",
            "loss_rpn_box_reg": "train/rpn_box_loss",
        }
        
        total_loss = 0.0
        for key, tensor in loss_dict.items():
            loss_val = tensor.item() if isinstance(tensor, torch.Tensor) else tensor
            total_loss += loss_val
            
            # Map to standard name or use original
            metric_name = loss_mapping.get(key, f"train/{key}")
            metrics[metric_name] = loss_val
        
        metrics["train/total_loss"] = total_loss
        
        wandb.log(metrics, step=self._step)
    
    def log_validation_metrics(
        self,
        val_loss: float,
        val_metrics: Optional[Dict[str, float]] = None,
        epoch: Optional[int] = None,
    ):
        """
        Log validation metrics at the end of an epoch.
        
        Args:
            val_loss: Validation loss
            val_metrics: Additional validation metrics (mAP, etc.)
            epoch: Current epoch
        """
        if epoch is not None:
            self._epoch = epoch
        
        metrics = {
            "val/loss": val_loss,
            "val/epoch": self._epoch,
        }
        
        if val_metrics:
            for key, value in val_metrics.items():
                metrics[f"val/{key}"] = value
        
        wandb.log(metrics, step=self._step)
    
    # =========================================================================
    # Gradient Norm Logging
    # =========================================================================
    
    def log_gradient_norms(self, model: nn.Module, step: Optional[int] = None):
        """
        Log gradient norms for CBAM attention layers and custom RoI head layers.
        
        This helps verify that custom layers are learning properly.
        
        Args:
            model: The model to analyze
            step: Global step number
        """
        if not self.config.log_gradients:
            return
        
        if step is not None:
            self._step = step
            
        if not self.should_log(self._step):
            return
        
        grad_norms = compute_gradient_norms(model)
        
        wandb.log(grad_norms, step=self._step)
    
    # =========================================================================
    # Validation Visualization
    # =========================================================================
    
    def set_validation_images(self, indices: List[int]):
        """
        Set fixed validation image indices for consistent visualization.
        
        Args:
            indices: List of dataset indices to use for visualization
        """
        self._val_image_indices = indices
    
    def log_validation_predictions(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        predictions: List[Dict[str, torch.Tensor]],
        epoch: Optional[int] = None,
    ):
        """
        Log validation predictions with ground truth comparison.
        
        Args:
            images: List of image tensors
            targets: List of ground truth target dictionaries
            predictions: List of prediction dictionaries
            epoch: Current epoch
        """
        if not self.config.log_images:
            return
        
        if epoch is not None:
            self._epoch = epoch
        
        wandb_images = []
        
        for idx, (img, target, pred) in enumerate(zip(images, targets, predictions)):
            wandb_img = create_wandb_image(
                img, target, pred,
                conf_threshold=self.config.conf_threshold,
                class_labels=ISAID_CLASS_LABELS,
            )
            wandb_images.append(wandb_img)
        
        wandb.log({
            "val/predictions": wandb_images,
            "val/epoch": self._epoch,
        }, step=self._step)
    
    # =========================================================================
    # Model Checkpointing
    # =========================================================================
    
    def log_model_checkpoint(
        self,
        model_path: str,
        model_name: str = "isaid-model",
        metadata: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ):
        """
        Log model checkpoint as W&B artifact.
        
        Args:
            model_path: Path to the .pth file
            model_name: Name for the artifact
            metadata: Additional metadata to attach
            aliases: Aliases for this version (e.g., ["best", "latest"])
        """
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=f"iSAID Mask R-CNN model checkpoint (epoch {self._epoch})",
            metadata=metadata or {
                "epoch": self._epoch,
                "step": self._step,
                **self.hyperparameters,
            },
        )
        
        artifact.add_file(model_path)
        
        aliases = aliases or ["latest"]
        self.run.log_artifact(artifact, aliases=aliases)
        
        print(f"Model checkpoint logged as artifact: {model_name}")
    
    def log_best_model(self, model_path: str, val_loss: float, val_map: Optional[float] = None):
        """
        Log the best model with "best" alias.
        
        Args:
            model_path: Path to best model .pth file
            val_loss: Validation loss of best model
            val_map: Optional mAP score
        """
        metadata = {
            "val_loss": val_loss,
            "epoch": self._epoch,
        }
        if val_map is not None:
            metadata["val_map"] = val_map
        
        self.log_model_checkpoint(
            model_path,
            model_name="isaid-model",
            metadata=metadata,
            aliases=["best", "latest"],
        )
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def finish(self):
        """Finish the W&B run."""
        if self._initialized:
            wandb.finish()
            print("W&B run finished.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


# =============================================================================
# Helper Functions
# =============================================================================

def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute L2 gradient norms for CBAM and RoI head layers.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with gradient norm values
    """
    grad_norms = {}
    
    cbam_grads = []
    roi_head_grads = []
    backbone_grads = []
    total_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grads.append(grad_norm)
            
            name_lower = name.lower()
            
            # CBAM attention layers
            if "cbam" in name_lower or "channel_attention" in name_lower or "spatial_attention" in name_lower:
                cbam_grads.append(grad_norm)
            
            # RoI head layers (box_head, mask_head, box_predictor, mask_predictor)
            elif any(x in name_lower for x in ["roi_heads", "box_head", "mask_head", "box_predictor", "mask_predictor"]):
                roi_head_grads.append(grad_norm)
            
            # Backbone layers
            elif "backbone" in name_lower:
                backbone_grads.append(grad_norm)
    
    # Compute aggregate norms
    if cbam_grads:
        grad_norms["grads/cbam_norm"] = np.sqrt(sum(g**2 for g in cbam_grads))
        grad_norms["grads/cbam_mean"] = np.mean(cbam_grads)
    
    if roi_head_grads:
        grad_norms["grads/roi_head_norm"] = np.sqrt(sum(g**2 for g in roi_head_grads))
        grad_norms["grads/roi_head_mean"] = np.mean(roi_head_grads)
    
    if backbone_grads:
        grad_norms["grads/backbone_norm"] = np.sqrt(sum(g**2 for g in backbone_grads))
    
    if total_grads:
        grad_norms["grads/total_norm"] = np.sqrt(sum(g**2 for g in total_grads))
    
    return grad_norms


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize image tensor and convert to numpy array.
    
    Args:
        image: Image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Numpy array [H, W, C] in range [0, 255] uint8
    """
    if isinstance(image, torch.Tensor):
        img = image.cpu().clone()
        
        # Denormalize if needed
        if img.min() < 0 or img.max() > 1:
            for c, (m, s) in enumerate(zip(mean, std)):
                img[c] = img[c] * s + m
        
        # Convert to numpy [H, W, C]
        img = img.permute(1, 2, 0).numpy()
        
        # Clip and convert to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        img = np.array(image)
    
    return img


def create_instance_mask(
    masks: torch.Tensor,
    labels: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Create a combined instance segmentation mask with class IDs.
    
    Args:
        masks: Binary masks [N, H, W] or [N, 1, H, W]
        labels: Class labels [N]
        height: Output height
        width: Output width
        
    Returns:
        Combined mask [H, W] with class IDs
    """
    combined_mask = np.zeros((height, width), dtype=np.int32)
    
    if len(masks) == 0:
        return combined_mask
    
    masks_np = masks.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Handle different mask shapes
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    
    # Overlay masks (later masks overwrite earlier ones)
    for mask, label in zip(masks_np, labels_np):
        # Resize mask if needed
        if mask.shape != (height, width):
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask > 0.5).astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
            mask = np.array(mask_pil) > 128
        else:
            mask = mask > 0.5
        
        combined_mask[mask] = int(label)
    
    return combined_mask


def create_wandb_image(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    prediction: Dict[str, torch.Tensor],
    conf_threshold: float = 0.5,
    class_labels: Dict[int, str] = None,
) -> wandb.Image:
    """
    Create a W&B Image with ground truth and prediction masks.
    
    Args:
        image: Image tensor [C, H, W]
        target: Ground truth dictionary with 'masks', 'labels', 'boxes'
        prediction: Prediction dictionary with 'masks', 'labels', 'boxes', 'scores'
        conf_threshold: Confidence threshold for predictions
        class_labels: Dictionary mapping class IDs to names
        
    Returns:
        wandb.Image with masks overlay
    """
    if class_labels is None:
        class_labels = ISAID_CLASS_LABELS
    
    # Denormalize image
    img_np = denormalize_image(image)
    height, width = img_np.shape[:2]
    
    # Create ground truth mask
    gt_masks = target.get("masks", torch.tensor([]))
    gt_labels = target.get("labels", torch.tensor([]))
    gt_mask = create_instance_mask(gt_masks, gt_labels, height, width)
    
    # Filter predictions by confidence
    pred_scores = prediction.get("scores", torch.tensor([]))
    if len(pred_scores) > 0:
        keep = pred_scores > conf_threshold
        pred_masks = prediction.get("masks", torch.tensor([]))[keep]
        pred_labels = prediction.get("labels", torch.tensor([]))[keep]
    else:
        pred_masks = torch.tensor([])
        pred_labels = torch.tensor([])
    
    pred_mask = create_instance_mask(pred_masks, pred_labels, height, width)
    
    # Create wandb image with masks
    masks_dict = {}
    
    if gt_mask.any():
        masks_dict["ground_truth"] = {
            "mask_data": gt_mask,
            "class_labels": class_labels,
        }
    
    if pred_mask.any():
        masks_dict["predictions"] = {
            "mask_data": pred_mask,
            "class_labels": class_labels,
        }
    
    # Create boxes annotations for wandb
    boxes_data = []
    
    # Ground truth boxes
    gt_boxes = target.get("boxes", torch.tensor([]))
    if len(gt_boxes) > 0:
        for box, label in zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy()):
            x1, y1, x2, y2 = box
            boxes_data.append({
                "position": {
                    "minX": float(x1) / width,
                    "maxX": float(x2) / width,
                    "minY": float(y1) / height,
                    "maxY": float(y2) / height,
                },
                "class_id": int(label),
                "box_caption": f"GT: {class_labels.get(int(label), f'Class {label}')}",
                "domain": "pixel",
            })
    
    # Prediction boxes
    pred_boxes = prediction.get("boxes", torch.tensor([]))
    if len(pred_scores) > 0:
        pred_boxes = pred_boxes[keep]
        pred_scores_filtered = pred_scores[keep]
        
        for box, label, score in zip(
            pred_boxes.cpu().numpy(),
            pred_labels.cpu().numpy(),
            pred_scores_filtered.cpu().numpy()
        ):
            x1, y1, x2, y2 = box
            boxes_data.append({
                "position": {
                    "minX": float(x1) / width,
                    "maxX": float(x2) / width,
                    "minY": float(y1) / height,
                    "maxY": float(y2) / height,
                },
                "class_id": int(label),
                "box_caption": f"Pred: {class_labels.get(int(label), f'Class {label}')} ({score:.2f})",
                "domain": "pixel",
                "scores": {"confidence": float(score)},
            })
    
    return wandb.Image(
        img_np,
        masks=masks_dict if masks_dict else None,
        boxes={
            "predictions": {
                "box_data": boxes_data,
                "class_labels": class_labels,
            }
        } if boxes_data else None,
    )


def get_fixed_val_batch(
    val_dataset,
    indices: List[int],
    device: str = "cuda",
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Get a fixed batch of validation images for consistent visualization.
    
    Args:
        val_dataset: Validation dataset
        indices: List of indices to sample
        device: Device to move tensors to
        
    Returns:
        Tuple of (images, targets)
    """
    images = []
    targets = []
    
    for idx in indices:
        img, target = val_dataset[idx]
        images.append(img)
        targets.append(target)
    
    return images, targets


# =============================================================================
# Integration with Trainer
# =============================================================================

def create_wandb_logger(
    hyperparameters: Dict[str, Any],
    project: str = "isaid-custom-segmentation",
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    log_freq: int = 20,
    num_val_images: int = 4,
) -> WandbLogger:
    """
    Create a WandbLogger with default configuration.
    
    Args:
        hyperparameters: Training hyperparameters dictionary
        project: W&B project name
        run_name: Optional run name
        tags: Optional list of tags
        log_freq: How often to log (every N steps)
        num_val_images: Number of validation images to visualize
        
    Returns:
        Configured WandbLogger
    """
    config = WandbConfig(
        project=project,
        run_name=run_name,
        tags=tags or [],
        log_freq=log_freq,
        num_val_images=num_val_images,
    )
    
    return WandbLogger(config, hyperparameters)


# Example hyperparameters structure
EXAMPLE_HYPERPARAMETERS = {
    # Training
    "learning_rate": 0.005,
    "batch_size": 2,
    "num_epochs": 50,
    "weight_decay": 0.0005,
    "momentum": 0.9,
    
    # Model
    "num_classes": 16,
    "backbone": "efficientnet_b0",
    "pretrained_backbone": True,
    
    # Custom layers
    "cbam_reduction_ratio": 16,
    "roi_head_layers": 4,
    
    # RPN
    "anchor_sizes": ((8, 16), (16, 32), (32, 64), (64, 128)),
    "aspect_ratios": ((0.5, 1.0, 2.0),) * 4,
    
    # Data
    "image_size": 800,
    "num_workers": 4,
    
    # Scheduler
    "scheduler_type": "onecycle",
    "max_lr": 0.01,
}


if __name__ == "__main__":
    # Test the logger
    print("Testing WandbLogger...")
    
    # Create mock hyperparameters
    hyperparams = EXAMPLE_HYPERPARAMETERS.copy()
    
    # Test without actual wandb (dry run)
    print(f"\nClass labels: {ISAID_CLASS_LABELS}")
    print(f"\nExample hyperparameters: {hyperparams}")
    
    # Test gradient norm computation on a simple model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 10)
            self.cbam = nn.Linear(10, 10)
            self.roi_heads = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.roi_heads(self.cbam(self.backbone(x)))
    
    model = MockModel()
    x = torch.randn(1, 10)
    y = model(x)
    y.sum().backward()
    
    grads = compute_gradient_norms(model)
    print(f"\nGradient norms: {grads}")
    
    print("\nWandbLogger module loaded successfully!")
