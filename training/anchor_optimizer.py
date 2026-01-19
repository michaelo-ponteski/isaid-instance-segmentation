"""
Anchor Size Optimizer for RPN using Optuna.
Optimizes anchor sizes based on dataset object statistics and validation performance.
"""

import os
import gc
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # Type hint fallback


@dataclass
class AnchorConfig:
    """Configuration for anchor sizes and aspect ratios."""
    sizes: Tuple[Tuple[int, ...], ...]
    aspect_ratios: Tuple[Tuple[float, ...], ...]
    
    def __post_init__(self):
        # Ensure aspect_ratios has same length as sizes
        if len(self.aspect_ratios) == 1 and len(self.sizes) > 1:
            self.aspect_ratios = self.aspect_ratios * len(self.sizes)
    
    def to_dict(self) -> Dict:
        return {
            'rpn_anchor_sizes': self.sizes,
            'rpn_aspect_ratios': self.aspect_ratios
        }


class DatasetAnchorAnalyzer:
    """
    Analyzes dataset to compute optimal anchor sizes based on object statistics.
    This provides a data-driven initialization for anchor optimization.
    """
    
    def __init__(self, dataset, num_samples: int = 1000):
        """
        Args:
            dataset: Dataset with targets containing 'boxes' key
            num_samples: Number of samples to analyze
        """
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        
    def compute_box_statistics(self) -> Dict[str, np.ndarray]:
        """Compute statistics of bounding boxes in the dataset."""
        widths = []
        heights = []
        aspect_ratios = []
        areas = []
        
        indices = np.random.choice(len(self.dataset), self.num_samples, replace=False)
        
        print(f"Analyzing {self.num_samples} samples for anchor statistics...")
        for idx in tqdm(indices, desc="Analyzing boxes"):
            try:
                _, target = self.dataset[idx]
                boxes = target.get('boxes', None)
                
                if boxes is None or len(boxes) == 0:
                    continue
                    
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.numpy()
                
                # boxes format: [x1, y1, x2, y2]
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                
                # Filter out invalid boxes
                valid = (w > 0) & (h > 0)
                w, h = w[valid], h[valid]
                
                if len(w) == 0:
                    continue
                    
                widths.extend(w.tolist())
                heights.extend(h.tolist())
                aspect_ratios.extend((w / h).tolist())
                areas.extend((w * h).tolist())
                
            except Exception as e:
                continue
        
        return {
            'widths': np.array(widths),
            'heights': np.array(heights),
            'aspect_ratios': np.array(aspect_ratios),
            'areas': np.array(areas)
        }
    
    def suggest_anchor_sizes(self, num_scales: int = 4) -> Tuple[Tuple[int, ...], ...]:
        """
        Suggest anchor sizes based on dataset statistics using k-means clustering.
        
        Args:
            num_scales: Number of feature pyramid levels
            
        Returns:
            Tuple of anchor size tuples for each pyramid level
        """
        stats = self.compute_box_statistics()
        areas = stats['areas']
        
        if len(areas) == 0:
            print("Warning: No valid boxes found, using default anchors")
            return ((16, 24), (32, 48), (64, 96), (128, 192))
        
        # Compute scale (sqrt of area) for clustering
        scales = np.sqrt(areas)
        
        # Use percentiles to determine anchor sizes
        percentiles = np.linspace(10, 90, num_scales * 2)
        scale_percentiles = np.percentile(scales, percentiles)
        
        # Group into pairs for each pyramid level
        anchor_sizes = []
        for i in range(num_scales):
            size1 = int(np.round(scale_percentiles[i * 2]))
            size2 = int(np.round(scale_percentiles[i * 2 + 1]))
            # Ensure minimum size and proper ordering
            size1 = max(8, size1)
            size2 = max(size1 + 4, size2)
            anchor_sizes.append((size1, size2))
        
        print(f"Suggested anchor sizes based on data: {anchor_sizes}")
        return tuple(anchor_sizes)
    
    def suggest_aspect_ratios(self, num_ratios: int = 3) -> Tuple[float, ...]:
        """
        Suggest aspect ratios based on dataset statistics.
        
        Args:
            num_ratios: Number of aspect ratios to suggest
            
        Returns:
            Tuple of aspect ratios
        """
        stats = self.compute_box_statistics()
        ratios = stats['aspect_ratios']
        
        if len(ratios) == 0:
            return (0.5, 1.0, 2.0)
        
        # Use percentiles for aspect ratios
        percentiles = np.linspace(20, 80, num_ratios)
        suggested = np.percentile(ratios, percentiles)
        
        # Round to reasonable values
        suggested = tuple(round(r, 2) for r in suggested)
        print(f"Suggested aspect ratios based on data: {suggested}")
        return suggested


class AnchorOptimizer:
    """
    Optuna-based anchor size optimizer for RPN.
    Optimizes anchor configurations by evaluating RPN recall on validation set.
    """
    
    def __init__(
        self,
        train_dataset,
        val_dataset,
        num_classes: int = 16,
        device: str = "cuda",
        num_fpn_levels: int = 4,
        base_aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    ):
        """
        Args:
            train_dataset: Training dataset for model fitting
            val_dataset: Validation dataset for evaluation
            num_classes: Number of classes including background
            device: Device to run optimization on
            num_fpn_levels: Number of FPN levels (anchor size groups)
            base_aspect_ratios: Base aspect ratios to use
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for anchor optimization. "
                "Install with: pip install optuna"
            )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_fpn_levels = num_fpn_levels
        self.base_aspect_ratios = base_aspect_ratios
        
        # Analyze dataset for informed search space
        self.analyzer = DatasetAnchorAnalyzer(train_dataset, num_samples=500)
        self.data_suggested_sizes = self.analyzer.suggest_anchor_sizes(num_fpn_levels)
        self.data_suggested_ratios = self.analyzer.suggest_aspect_ratios()
        
    def _create_model_with_anchors(self, anchor_config: AnchorConfig):
        """Create a model with specified anchor configuration."""
        from models.maskrcnn_model import CustomMaskRCNN
        
        model = CustomMaskRCNN(
            num_classes=self.num_classes,
            pretrained_backbone=True,
            **anchor_config.to_dict()
        )
        model.to(self.device)
        return model
    
    def _compute_rpn_recall(
        self,
        model,
        dataloader,
        iou_thresholds: List[float] = [0.5, 0.75],
        max_batches: int = 50
    ) -> Dict[str, float]:
        """
        Compute RPN recall at different IoU thresholds.
        
        Args:
            model: Model to evaluate
            dataloader: Validation dataloader
            iou_thresholds: IoU thresholds for recall computation
            max_batches: Maximum batches to evaluate
            
        Returns:
            Dictionary with recall values
        """
        from torchvision.ops import box_iou
        
        model.eval()
        recalls = {f"recall@{t}": [] for t in iou_thresholds}
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                try:
                    images = [img.to(self.device) for img in images]
                    
                    # Get proposals from RPN
                    if isinstance(images, list):
                        images_tensor = torch.stack(images)
                    else:
                        images_tensor = images
                    
                    from torchvision.models.detection.image_list import ImageList
                    original_sizes = [img.shape[-2:] for img in images]
                    image_list = ImageList(images_tensor, original_sizes)
                    
                    features = model.backbone(images_tensor)
                    proposals, _ = model.rpn(image_list, features, None)
                    
                    # Compute recall for each image
                    for props, target in zip(proposals, targets):
                        gt_boxes = target['boxes'].to(self.device)
                        
                        if len(gt_boxes) == 0:
                            continue
                        
                        if len(props) == 0:
                            for t in iou_thresholds:
                                recalls[f"recall@{t}"].append(0.0)
                            continue
                        
                        ious = box_iou(props, gt_boxes)
                        max_ious, _ = ious.max(dim=0)
                        
                        for t in iou_thresholds:
                            recall = (max_ious >= t).float().mean().item()
                            recalls[f"recall@{t}"].append(recall)
                    
                    # Clear memory
                    del images, images_tensor, features, proposals
                    
                except Exception as e:
                    continue
        
        # Average recalls
        return {k: np.mean(v) if v else 0.0 for k, v in recalls.items()}
    
    def _quick_train(
        self,
        model,
        dataloader,
        num_iterations: int = 100,
        lr: float = 0.001
    ):
        """Quick training to adapt model before evaluation."""
        model.train()
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, momentum=0.9
        )
        
        iteration = 0
        for images, targets in dataloader:
            if iteration >= num_iterations:
                break
            
            try:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Skip if no valid targets
                if all(len(t['boxes']) == 0 for t in targets):
                    continue
                
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                iteration += 1
                
                del images, targets, loss_dict, loss
                
            except Exception:
                continue
        
        torch.cuda.empty_cache()
    
    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function for anchor optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Negative recall (for minimization)
        """
        # Sample anchor sizes for each FPN level
        anchor_sizes = []
        
        for level in range(self.num_fpn_levels):
            # Use data-suggested values as center of search range
            base_small = self.data_suggested_sizes[level][0]
            base_large = self.data_suggested_sizes[level][1]
            
            # Define search range around suggested values
            small_min = max(8, int(base_small * 0.5))
            small_max = int(base_small * 1.5)
            large_min = max(small_min + 4, int(base_large * 0.5))
            large_max = int(base_large * 1.5)
            
            size_small = trial.suggest_int(f"size_l{level}_small", small_min, small_max, step=4)
            size_large = trial.suggest_int(f"size_l{level}_large", large_min, large_max, step=4)
            
            # Ensure ordering
            if size_large <= size_small:
                size_large = size_small + 8
            
            anchor_sizes.append((size_small, size_large))
        
        # Optionally optimize aspect ratios
        optimize_ratios = trial.suggest_categorical("optimize_ratios", [True, False])
        
        if optimize_ratios:
            ratio_1 = trial.suggest_float("ratio_1", 0.3, 0.7, step=0.1)
            ratio_2 = trial.suggest_float("ratio_2", 0.8, 1.2, step=0.1)
            ratio_3 = trial.suggest_float("ratio_3", 1.5, 3.0, step=0.25)
            aspect_ratios = ((ratio_1, ratio_2, ratio_3),) * self.num_fpn_levels
        else:
            aspect_ratios = (self.base_aspect_ratios,) * self.num_fpn_levels
        
        anchor_config = AnchorConfig(
            sizes=tuple(anchor_sizes),
            aspect_ratios=aspect_ratios
        )
        
        print(f"\nTrial {trial.number}: Testing anchors {anchor_sizes}")
        
        try:
            # Create model with trial anchors
            model = self._create_model_with_anchors(anchor_config)
            
            # Create dataloaders
            def collate_fn(batch):
                return tuple(zip(*batch))
            
            train_subset = Subset(
                self.train_dataset, 
                range(min(200, len(self.train_dataset)))
            )
            val_subset = Subset(
                self.val_dataset,
                range(min(100, len(self.val_dataset)))
            )
            
            train_loader = DataLoader(
                train_subset, batch_size=2, shuffle=True,
                collate_fn=collate_fn, num_workers=0
            )
            val_loader = DataLoader(
                val_subset, batch_size=1, shuffle=False,
                collate_fn=collate_fn, num_workers=0
            )
            
            # Quick training
            self._quick_train(model, train_loader, num_iterations=50)
            
            # Evaluate RPN recall
            recalls = self._compute_rpn_recall(model, val_loader, max_batches=30)
            
            # Combined metric (weighted average of recalls)
            score = 0.5 * recalls.get("recall@0.5", 0) + 0.5 * recalls.get("recall@0.75", 0)
            
            print(f"  Recalls: {recalls}, Score: {score:.4f}")
            
            # Cleanup
            del model, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            
            return -score  # Negative because Optuna minimizes
            
        except Exception as e:
            print(f"  Trial failed: {e}")
            return 0.0  # Return worst score on failure
    
    def optimize(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        study_name: str = "anchor_optimization",
        storage: Optional[str] = None,
    ) -> AnchorConfig:
        """
        Run anchor optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Name for the Optuna study
            storage: Optional database URL for persistence
            
        Returns:
            Best anchor configuration found
        """
        print("=" * 60)
        print("Starting Anchor Size Optimization with Optuna")
        print("=" * 60)
        print(f"Data-suggested anchor sizes: {self.data_suggested_sizes}")
        print(f"Data-suggested aspect ratios: {self.data_suggested_ratios}")
        print(f"Running {n_trials} trials...")
        print("=" * 60)
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Extract best configuration
        best_trial = study.best_trial
        print("\n" + "=" * 60)
        print("Optimization Complete!")
        print("=" * 60)
        print(f"Best trial: {best_trial.number}")
        print(f"Best score: {-best_trial.value:.4f}")
        print(f"Best params: {best_trial.params}")
        
        # Reconstruct best anchor config
        anchor_sizes = []
        for level in range(self.num_fpn_levels):
            size_small = best_trial.params[f"size_l{level}_small"]
            size_large = best_trial.params[f"size_l{level}_large"]
            anchor_sizes.append((size_small, size_large))
        
        if best_trial.params.get("optimize_ratios", False):
            aspect_ratios = ((
                best_trial.params["ratio_1"],
                best_trial.params["ratio_2"],
                best_trial.params["ratio_3"],
            ),) * self.num_fpn_levels
        else:
            aspect_ratios = (self.base_aspect_ratios,) * self.num_fpn_levels
        
        best_config = AnchorConfig(
            sizes=tuple(anchor_sizes),
            aspect_ratios=aspect_ratios
        )
        
        print(f"\nBest anchor configuration:")
        print(f"  Sizes: {best_config.sizes}")
        print(f"  Aspect ratios: {best_config.aspect_ratios}")
        
        return best_config


def optimize_anchors_for_dataset(
    data_root: str,
    num_classes: int = 16,
    n_trials: int = 20,
    device: str = "cuda",
    image_size: int = 800,
) -> AnchorConfig:
    """
    Convenience function to run anchor optimization on a dataset.
    
    Args:
        data_root: Root directory of the dataset
        num_classes: Number of classes
        n_trials: Number of optimization trials
        device: Device to use
        image_size: Image size for the dataset
        
    Returns:
        Best anchor configuration
    """
    from datasets.isaid_dataset import iSAIDDataset
    from training.transforms import get_transforms
    
    print("Loading datasets for anchor optimization...")
    train_dataset = iSAIDDataset(
        data_root, split="train",
        transforms=get_transforms(train=False),  # No augmentation for analysis
        image_size=image_size
    )
    val_dataset = iSAIDDataset(
        data_root, split="val",
        transforms=get_transforms(train=False),
        image_size=image_size
    )
    
    optimizer = AnchorOptimizer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_classes=num_classes,
        device=device,
    )
    
    return optimizer.optimize(n_trials=n_trials)


def analyze_dataset_anchors(
    data_root: str,
    image_size: int = 800,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Analyze dataset to get suggested anchor sizes without optimization.
    
    Args:
        data_root: Root directory of the dataset
        image_size: Image size
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary with suggested configurations and statistics
    """
    from datasets.isaid_dataset import iSAIDDataset
    from training.transforms import get_transforms
    
    dataset = iSAIDDataset(
        data_root, split="train",
        transforms=get_transforms(train=False),
        image_size=image_size
    )
    
    analyzer = DatasetAnchorAnalyzer(dataset, num_samples=num_samples)
    stats = analyzer.compute_box_statistics()
    suggested_sizes = analyzer.suggest_anchor_sizes()
    suggested_ratios = analyzer.suggest_aspect_ratios()
    
    return {
        'suggested_sizes': suggested_sizes,
        'suggested_ratios': suggested_ratios,
        'statistics': {
            'mean_width': float(np.mean(stats['widths'])),
            'mean_height': float(np.mean(stats['heights'])),
            'mean_area': float(np.mean(stats['areas'])),
            'std_area': float(np.std(stats['areas'])),
            'min_area': float(np.min(stats['areas'])),
            'max_area': float(np.max(stats['areas'])),
            'mean_aspect_ratio': float(np.mean(stats['aspect_ratios'])),
            'num_boxes_analyzed': len(stats['widths']),
        }
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize anchor sizes for RPN")
    parser.add_argument("--data_root", type=str, default="iSAID_patches",
                       help="Dataset root directory")
    parser.add_argument("--n_trials", type=int, default=20,
                       help="Number of optimization trials")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze dataset, don't optimize")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        results = analyze_dataset_anchors(args.data_root)
        print("\nDataset Analysis Results:")
        print(f"Suggested sizes: {results['suggested_sizes']}")
        print(f"Suggested ratios: {results['suggested_ratios']}")
        print(f"Statistics: {results['statistics']}")
    else:
        best_config = optimize_anchors_for_dataset(
            data_root=args.data_root,
            n_trials=args.n_trials,
            device=args.device,
        )
        print(f"\nUse these in your model:")
        print(f"rpn_anchor_sizes={best_config.sizes}")
        print(f"rpn_aspect_ratios={best_config.aspect_ratios}")
