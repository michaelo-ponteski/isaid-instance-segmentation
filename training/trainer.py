"""
Training loop for Mask R-CNN (memory-safe version).
"""

import os
import time
import gc
import math
from pathlib import Path
from typing import Optional, Tuple, Union, Literal

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    StepLR, 
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR,
)
from tqdm.auto import tqdm

from models.maskrcnn_model import CustomMaskRCNN, get_custom_maskrcnn, get_custom_maskrcnn_with_optimized_anchors
from datasets.isaid_dataset import iSAIDDataset
from training.transforms import get_transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def collate_fn(batch):
    return tuple(zip(*batch))


def create_datasets(
    data_root: str,
    image_size: int = 800,
    subset_fraction: float = 1.0,
) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets.
    
    This is a helper function to create datasets once and reuse them
    across multiple training runs with different models/backbones.
    
    Args:
        data_root: Path to the dataset root directory
        image_size: Image size for resizing
        subset_fraction: Fraction of data to use (0.0 to 1.0)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("Loading datasets...")
    train_dataset_full = iSAIDDataset(
        data_root,
        split="train",
        transforms=get_transforms(train=True),
        image_size=image_size,
    )
    val_dataset_full = iSAIDDataset(
        data_root,
        split="val",
        transforms=get_transforms(train=False),
        image_size=image_size,
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
    Trainer class for Mask R-CNN models.
    
    Can be initialized in two ways:
    1. With pre-created datasets and model (recommended for multiple runs)
    2. With data_root to create datasets internally (legacy mode)
    
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
        lr: float = 0.0001,
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
        # Learning rate scheduler options
        scheduler_type: Literal["step", "cosine", "onecycle", "plateau", "warmup_cosine"] = "onecycle",
        warmup_epochs: int = 1,
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
            scheduler_type: LR scheduler type - "step", "cosine", "onecycle", "plateau", "warmup_cosine"
            warmup_epochs: Number of warmup epochs (for warmup_cosine scheduler)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.batch_size = batch_size
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs

        # Handle datasets
        if train_dataset is not None and val_dataset is not None:
            # Use provided datasets
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            print(f"Using provided datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")
        elif data_root is not None:
            # Legacy mode: create datasets from data_root
            self.train_dataset, self.val_dataset = create_datasets(
                data_root=data_root,
                image_size=image_size,
                subset_fraction=subset_fraction,
            )
            self.data_root = data_root
        else:
            raise ValueError(
                "Either provide (train_dataset, val_dataset) or data_root"
            )

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

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
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
            raise ValueError(
                "Either provide a model or data_root to create one"
            )
        
        self.model.to(self.device)

        # Setup optimizer and scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        # Scheduler will be created in fit() when we know epochs
        self.scheduler = None
        self._scheduler_type = scheduler_type

        self.scaler = GradScaler(enabled=self.use_amp)

        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def _create_scheduler(self, epochs: int):
        """Create learning rate scheduler based on scheduler_type."""
        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch
        
        if self._scheduler_type == "step":
            # Classic step decay
            self.scheduler = StepLR(
                self.optimizer, step_size=max(1, epochs // 3), gamma=0.1
            )
            self._scheduler_step_per_batch = False
            print(f"Using StepLR scheduler (step every {epochs // 3} epochs)")
            
        elif self._scheduler_type == "cosine":
            # Cosine annealing with warm restarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(1, epochs // 4), T_mult=2, eta_min=self.lr * 0.01
            )
            self._scheduler_step_per_batch = False
            print(f"Using CosineAnnealingWarmRestarts scheduler")
            
        elif self._scheduler_type == "onecycle":
            # OneCycleLR - typically the best for fast convergence
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.lr * 10,  # Peak LR is 10x the base
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                div_factor=10,  # Initial LR = max_lr / 10
                final_div_factor=100,  # Final LR = max_lr / 1000
            )
            self._scheduler_step_per_batch = True
            print(f"Using OneCycleLR scheduler (max_lr={self.lr * 10:.6f})")
            
        elif self._scheduler_type == "plateau":
            # Reduce on plateau - reactive to validation loss
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3,
                min_lr=self.lr * 0.001, verbose=True
            )
            self._scheduler_step_per_batch = False
            self._scheduler_needs_val_loss = True
            print(f"Using ReduceLROnPlateau scheduler")
            
        elif self._scheduler_type == "warmup_cosine":
            # Linear warmup + cosine decay
            warmup_steps = self.warmup_epochs * steps_per_epoch
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            self._scheduler_step_per_batch = True
            print(f"Using Warmup+Cosine scheduler (warmup={self.warmup_epochs} epochs)")
        else:
            raise ValueError(f"Unknown scheduler type: {self._scheduler_type}")

    def find_lr(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_iter: int = 100,
        smooth_factor: float = 0.05,
        plot: bool = True,
    ) -> float:
        """
        Learning Rate Finder using the LR range test.
        
        Runs a few iterations with exponentially increasing LR to find the optimal range.
        
        Args:
            start_lr: Starting learning rate
            end_lr: Maximum learning rate to test
            num_iter: Number of iterations
            smooth_factor: Smoothing factor for loss curve
            plot: Whether to plot the results
            
        Returns:
            Suggested learning rate (point of steepest descent)
        """
        print("=" * 60)
        print("Learning Rate Finder")
        print("=" * 60)
        
        # Save initial state
        initial_state = {
            'model': {k: v.clone() for k, v in self.model.state_dict().items()},
            'optimizer': {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in self.optimizer.state_dict().items()},
        }
        
        # Setup
        self.model.train()
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        
        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        lrs = []
        losses = []
        smoothed_loss = 0
        best_loss = float('inf')
        
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
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Skip empty batches
            if all(len(t['boxes']) == 0 for t in targets):
                continue
            
            self.optimizer.zero_grad()
            
            try:
                if self.use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        loss_dict = self.model(images, targets)
                        loss = sum(loss_dict.values())
                    
                    if not torch.isfinite(loss):
                        break
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())
                    
                    if not torch.isfinite(loss):
                        break
                    
                    loss.backward()
                    self.optimizer.step()
                
                loss_val = loss.item()
                
                # Smooth loss
                if iteration == 0:
                    smoothed_loss = loss_val
                else:
                    smoothed_loss = smooth_factor * loss_val + (1 - smooth_factor) * smoothed_loss
                
                # Stop if loss explodes
                if smoothed_loss > 4 * best_loss and iteration > 10:
                    print(f"\nStopping early - loss exploding at lr={lr:.2e}")
                    break
                
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss
                
                lrs.append(lr)
                losses.append(smoothed_loss)
                
                pbar.set_postfix(lr=f"{lr:.2e}", loss=f"{smoothed_loss:.4f}")
                
            except RuntimeError as e:
                print(f"\nError at lr={lr:.2e}: {e}")
                break
            
            # Update LR
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            del images, targets, loss, loss_dict
            torch.cuda.empty_cache()
        
        # Restore initial state
        self.model.load_state_dict(initial_state['model'])
        self.optimizer.load_state_dict(initial_state['optimizer'])
        
        # Find suggested LR (steepest descent point)
        if len(losses) < 10:
            print("Not enough data points collected. Using default LR.")
            return self.lr
        
        # Compute gradient of loss curve
        losses_arr = torch.tensor(losses)
        lrs_arr = torch.tensor(lrs)
        
        # Find the point with steepest negative gradient
        gradients = (losses_arr[1:] - losses_arr[:-1]) / (torch.log10(lrs_arr[1:]) - torch.log10(lrs_arr[:-1]))
        
        # Find minimum gradient (steepest descent)
        min_grad_idx = torch.argmin(gradients).item()
        suggested_lr = lrs[min_grad_idx]
        
        # Use a bit lower LR for stability (1/10th of the steepest point)
        suggested_lr = suggested_lr / 10
        
        print(f"\n{'=' * 60}")
        print(f"Suggested Learning Rate: {suggested_lr:.6f}")
        print(f"{'=' * 60}")
        
        if plot:
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(lrs, losses, 'b-', linewidth=2)
                ax.axvline(x=suggested_lr, color='r', linestyle='--', 
                          label=f'Suggested LR: {suggested_lr:.2e}')
                ax.axvline(x=suggested_lr * 10, color='orange', linestyle=':', 
                          label=f'Steepest point: {suggested_lr * 10:.2e}')
                ax.set_xscale('log')
                ax.set_xlabel('Learning Rate (log scale)')
                ax.set_ylabel('Loss (smoothed)')
                ax.set_title('Learning Rate Finder')
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
            param_group['lr'] = lr
        print(f"Learning rate set to: {lr:.6f}")

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loss_accumulator = {}

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
                        self.scaler.update()  # Still update scaler to prevent stall
                        del images, targets, loss, loss_dict
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients - replace NaN/Inf with zeros and clip
                    for param in self.model.parameters():
                        if param.grad is not None:
                            torch.nan_to_num_(
                                param.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
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

                    # Clip gradients - replace NaN/Inf with zeros and clip
                    for param in self.model.parameters():
                        if param.grad is not None:
                            torch.nan_to_num_(
                                param.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value
                # Accumulate individual losses
                for k, v in loss_dict.items():
                    loss_accumulator.setdefault(k, 0.0)
                    loss_accumulator[k] += v.item()
                # Show all losses in pbar with current LR
                loss_postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                loss_postfix["total"] = f"{loss_value:.4f}"
                loss_postfix["lr"] = f"{self.get_lr():.2e}"
                pbar.set_postfix(loss=loss_postfix)
                
                # Step scheduler per batch if needed (OneCycleLR, warmup_cosine)
                if self.scheduler is not None and getattr(self, '_scheduler_step_per_batch', False):
                    self.scheduler.step()
                    
            except RuntimeError as e:
                print(f"Warning: RuntimeError in batch: {e}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            del images, targets, loss, loss_dict
            torch.cuda.empty_cache()

        # Step scheduler per epoch if needed (StepLR, Cosine)
        if self.scheduler is not None and not getattr(self, '_scheduler_step_per_batch', False):
            if not getattr(self, '_scheduler_needs_val_loss', False):
                self.scheduler.step()
        
        # Return average losses
        avg_losses = {
            k: v / len(self.train_loader) for k, v in loss_accumulator.items()
        }
        avg_losses["total"] = total_loss / len(self.train_loader)
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
            torch.cuda.empty_cache()

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

    def fit(self, epochs=20, save_dir="checkpoints", find_lr_first=False):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            find_lr_first: If True, run LR finder before training
        """
        # Optionally find best LR
        if find_lr_first:
            suggested_lr = self.find_lr()
            self.set_lr(suggested_lr)
        
        # Create scheduler now that we know epochs
        self._create_scheduler(epochs)
        
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{epochs} | LR: {self.get_lr():.2e}")
            print(f"{'=' * 50}")

            start = time.time()
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate()
            epoch_time = time.time() - start

            # Print detailed losses
            print(f"\nEpoch {epoch} Results (Time: {epoch_time:.1f}s):")
            print(f"  Train Losses:")
            for k, v in train_losses.items():
                print(f"    {k}: {v:.4f}")
            print(f"  Val Losses:")
            for k, v in val_losses.items():
                print(f"    {k}: {v:.4f}")

            val_loss = val_losses["total"]
            
            # Step plateau scheduler with validation loss
            if self.scheduler is not None and getattr(self, '_scheduler_needs_val_loss', False):
                self.scheduler.step(val_loss)
            
            self.save_checkpoint(f"{save_dir}/last.pth", epoch, val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(f"{save_dir}/best.pth", epoch, val_loss)
                print("-> New best model saved")

            gc.collect()
            torch.cuda.empty_cache()

        print(f"\nTraining complete. Best val loss: {best_loss:.4f}")

    @torch.no_grad()
    def visualize_predictions(self, num_samples=5, score_threshold=0.5, save_path=None):
        """Visualize model predictions on validation set.

        Args:
            num_samples: Number of validation samples to visualize
            score_threshold: Minimum confidence score for predictions
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        self.model.eval()

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

            # Convert image for visualization
            if isinstance(image, torch.Tensor):
                img_display = image.permute(1, 2, 0).cpu().numpy()
            else:
                img_display = np.array(image)

            # Normalize if needed
            if img_display.max() <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)

            # Get predictions
            image_tensor = (
                image.unsqueeze(0).to(self.device)
                if len(image.shape) == 3
                else image.to(self.device)
            )
            predictions = self.model([image_tensor])[0]

            # Filter predictions by score
            keep_idx = predictions["scores"] > score_threshold
            pred_boxes = predictions["boxes"][keep_idx].cpu().numpy()
            pred_labels = predictions["labels"][keep_idx].cpu().numpy()
            pred_scores = predictions["scores"][keep_idx].cpu().numpy()

            # Ground truth visualization
            ax_gt = axes[idx, 0]
            ax_gt.imshow(img_display)
            ax_gt.set_title(f"Ground Truth (Sample {sample_idx})")
            ax_gt.axis("off")

            if "boxes" in target and len(target["boxes"]) > 0:
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                for box, label in zip(gt_boxes, gt_labels):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor="green",
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
                        bbox=dict(facecolor="green", alpha=0.7),
                    )

            # Predictions visualization
            ax_pred = axes[idx, 1]
            ax_pred.imshow(img_display)
            ax_pred.set_title(
                f"Predictions (Sample {sample_idx}, {len(pred_boxes)} detections)"
            )
            ax_pred.axis("off")

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
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
                    bbox=dict(facecolor="red", alpha=0.7),
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
