"""
Anchor Size Optimizer for RPN using Optuna.

This module optimizes anchor sizes using GEOMETRIC COVERAGE (theoretical recall)

"""

import os
import gc
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import torch
import numpy as np
from torchvision.ops import box_iou
from tqdm.auto import tqdm

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = Any  # Type hint fallback


# FPN stride configuration
FPN_STRIDES = {
    'P2': 4,
    'P3': 8,
    'P4': 16,
    'P5': 32,
}


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

    def __repr__(self):
        return f"AnchorConfig(sizes={self.sizes}, aspect_ratios={self.aspect_ratios})"


def generate_anchors_for_image(
    image_size: Tuple[int, int],
    anchor_sizes: Tuple[Tuple[int, ...], ...],
    aspect_ratios: Tuple[Tuple[float, ...], ...],
    strides: List[int] = [4, 8, 16, 32],
) -> torch.Tensor:
    """
    Generate all anchor boxes for a given image size.

    Args:
        image_size: (height, width) of the image
        anchor_sizes: Tuple of anchor sizes per FPN level
        aspect_ratios: Tuple of aspect ratios per FPN level
        strides: FPN strides for each level

    Returns:
        Tensor of shape [N, 4] with anchor boxes in (x1, y1, x2, y2) format
    """
    H, W = image_size
    all_anchors = []

    for level_idx, (sizes, ratios, stride) in enumerate(zip(anchor_sizes, aspect_ratios, strides)):
        # Feature map size at this level
        fH, fW = H // stride, W // stride

        # Generate grid centers
        shifts_x = torch.arange(0, fW) * stride + stride // 2
        shifts_y = torch.arange(0, fH) * stride + stride // 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        # Generate anchor shapes for this level
        for size in sizes:
            for ratio in ratios:
                # Anchor dimensions
                w = size * np.sqrt(ratio)
                h = size / np.sqrt(ratio)

                # Create anchors at all grid positions
                x1 = shift_x - w / 2
                y1 = shift_y - h / 2
                x2 = shift_x + w / 2
                y2 = shift_y + h / 2

                level_anchors = torch.stack([x1, y1, x2, y2], dim=1).float()
                all_anchors.append(level_anchors)

    return torch.cat(all_anchors, dim=0)


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
        self._cached_stats = None

    def compute_box_statistics(self) -> Dict[str, np.ndarray]:
        """Compute statistics of bounding boxes in the dataset."""
        if self._cached_stats is not None:
            return self._cached_stats

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

        self._cached_stats = {
            'widths': np.array(widths),
            'heights': np.array(heights),
            'aspect_ratios': np.array(aspect_ratios),
            'areas': np.array(areas)
        }
        return self._cached_stats

    def suggest_anchor_sizes_with_stride_constraints(
        self,
        strides: List[int] = [4, 8, 16, 32],
    ) -> Tuple[Tuple[int, ...], ...]:
        """
        Suggest anchor sizes respecting FPN stride constraints.

        Rule: Anchor size should be >= stride * 2 for effective detection.

        Args:
            strides: FPN strides for each level

        Returns:
            Tuple of anchor size tuples for each pyramid level
        """
        stats = self.compute_box_statistics()
        scales = np.sqrt(stats['areas'])

        if len(scales) == 0:
            print("Warning: No valid boxes found, using default anchors")
            return tuple((s*2, s*4) for s in strides)

        # Get scale distribution
        percentiles = [10, 25, 50, 75, 90, 95]
        scale_dist = {p: np.percentile(scales, p) for p in percentiles}

        print(f"\nObject scale distribution:")
        for p, v in scale_dist.items():
            print(f"  {p}th percentile: {v:.1f} px")

        anchor_sizes = []
        for i, stride in enumerate(strides):
            min_size = stride * 2  # Minimum valid anchor size
            max_size = stride * 12  # Maximum reasonable anchor size

            if i == 0:  # P2 - smallest objects
                target_percentile = 25
            elif i == 1:  # P3
                target_percentile = 50
            elif i == 2:  # P4
                target_percentile = 75
            else:  # P5 - largest objects
                target_percentile = 90

            data_suggested = scale_dist[target_percentile]

            # Apply stride constraints
            size1 = max(min_size, int(np.round(data_suggested * 0.75)))
            size2 = max(size1 + stride, min(max_size, int(np.round(data_suggested * 1.5))))

            anchor_sizes.append((size1, size2))

        print(f"\nStride-constrained anchor sizes: {anchor_sizes}")
        return tuple(anchor_sizes)
    

    def suggest_aspect_ratios(self, num_ratios: int = 3) -> Tuple[float, ...]:
        """
        Suggest aspect ratios based on dataset statistics.
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


class GeometricAnchorOptimizer:
    """
    Optuna-based anchor optimizer using GEOMETRIC COVERAGE.

    This optimizer evaluates anchor configurations by computing theoretical recall
    based purely on IoU between anchors and ground truth boxes
    """

    def __init__(
        self,
        dataset,
        image_size: Tuple[int, int] = (800, 800),
        strides: List[int] = [4, 8, 16, 32],
        base_aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        num_samples: int = 500,
    ):
        """
        Args:
            dataset: Dataset with targets containing 'boxes' key
            image_size: Image size (H, W) for anchor generation
            strides: FPN strides for each level
            base_aspect_ratios: Default aspect ratios
            num_samples: Number of samples to evaluate
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for anchor optimization. "
                "Install with: pip install optuna"
            )

        self.dataset = dataset
        self.image_size = image_size
        self.strides = strides
        self.num_fpn_levels = len(strides)
        self.base_aspect_ratios = base_aspect_ratios
        self.num_samples = min(num_samples, len(dataset))

        # Analyze dataset
        self.analyzer = DatasetAnchorAnalyzer(dataset, num_samples=num_samples)
        self.data_stats = self.analyzer.compute_box_statistics()
        self.suggested_ratios = self.analyzer.suggest_aspect_ratios()

        # Cache ground truth boxes for fast evaluation
        self._cache_gt_boxes()

    def _cache_gt_boxes(self):
        """Cache ground truth boxes from dataset for fast evaluation."""
        print(f"Caching GT boxes from {self.num_samples} samples...")
        self.gt_boxes_list = []

        indices = np.random.choice(len(self.dataset), self.num_samples, replace=False)

        for idx in tqdm(indices, desc="Caching GT boxes"):
            try:
                _, target = self.dataset[idx]
                boxes = target.get('boxes', None)

                if boxes is not None and len(boxes) > 0:
                    if isinstance(boxes, np.ndarray):
                        boxes = torch.from_numpy(boxes).float()
                    self.gt_boxes_list.append(boxes.float())
            except Exception:
                continue

        self.total_gt_boxes = sum(len(b) for b in self.gt_boxes_list)
        print(f"Cached {self.total_gt_boxes} GT boxes from {len(self.gt_boxes_list)} images")

    def compute_geometric_recall(
        self,
        anchor_sizes: Tuple[Tuple[int, ...], ...],
        aspect_ratios: Tuple[Tuple[float, ...], ...],
        iou_thresholds: List[float] = [0.5, 0.7],
    ) -> Dict[str, float]:
        """
        Compute geometric recall - the fraction of GT boxes that have at least
        one anchor with IoU >= threshold.

        This is THEORETICAL RECALL - the best possible recall if the model
        were perfectly trained.

        Args:
            anchor_sizes: Anchor sizes per FPN level
            aspect_ratios: Aspect ratios per FPN level
            iou_thresholds: IoU thresholds for recall computation

        Returns:
            Dictionary with recall values
        """
        # Generate all anchors for the image size
        anchors = generate_anchors_for_image(
            self.image_size, anchor_sizes, aspect_ratios, self.strides
        )

        recalls = {f"recall@{t}": 0 for t in iou_thresholds}
        matches = {f"recall@{t}": 0 for t in iou_thresholds}

        for gt_boxes in self.gt_boxes_list:
            if len(gt_boxes) == 0:
                continue

            # Compute IoU between all anchors and GT boxes
            ious = box_iou(anchors, gt_boxes)  # [num_anchors, num_gt]

            # For each GT box, find max IoU with any anchor
            max_ious, _ = ious.max(dim=0)  # [num_gt]

            for t in iou_thresholds:
                matches[f"recall@{t}"] += (max_ious >= t).sum().item()

        for t in iou_thresholds:
            recalls[f"recall@{t}"] = matches[f"recall@{t}"] / max(1, self.total_gt_boxes)

        return recalls

    def objective(self, trial: Trial) -> float:
        """
        Optuna objective function for anchor optimization.

        Optimizes geometric recall while respecting stride constraints.

        Args:
            trial: Optuna trial object

        Returns:
            Combined recall score (for maximization)
        """
        anchor_sizes = []

        for level_idx, stride in enumerate(self.strides):
            # Anchor size must be >= stride * 2 to be effective
            min_size = stride * 2
            max_size = stride * 16

            # Suggest sizes for this level
            size1 = trial.suggest_int(
                f"size_l{level_idx}_1",
                min_size,
                max_size,
                step=stride  # Step by stride for cleaner values
            )
            size2 = trial.suggest_int(
                f"size_l{level_idx}_2",
                min_size,
                max_size,
                step=stride
            )

            # Ensure size1 <= size2
            if size1 > size2:
                size1, size2 = size2, size1

            anchor_sizes.append((size1, size2))

        # Optionally optimize aspect ratios
        optimize_ratios = trial.suggest_categorical("optimize_ratios", [True, False])

        if optimize_ratios:
            ratio_1 = trial.suggest_float("ratio_1", 0.3, 0.8, step=0.1)
            ratio_2 = trial.suggest_float("ratio_2", 0.8, 1.2, step=0.1)
            ratio_3 = trial.suggest_float("ratio_3", 1.5, 3.0, step=0.25)
            aspect_ratios = ((ratio_1, ratio_2, ratio_3),) * self.num_fpn_levels
        else:
            aspect_ratios = (self.base_aspect_ratios,) * self.num_fpn_levels

        # Compute geometric recall
        recalls = self.compute_geometric_recall(
            tuple(anchor_sizes), aspect_ratios
        )

        # Combined score: weighted average of recalls
        score = 0.7 * recalls["recall@0.5"] + 0.3 * recalls["recall@0.7"]

        # Store actual config for later retrieval
        trial.set_user_attr("anchor_sizes", anchor_sizes)
        trial.set_user_attr("aspect_ratios", aspect_ratios)
        trial.set_user_attr("recalls", recalls)

        return score

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: str = "geometric_anchor_optimization",
    ) -> AnchorConfig:
        """
        Run anchor optimization.

        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Name for the Optuna study

        Returns:
            Best anchor configuration found
        """
        print("=" * 60)
        print("Geometric Anchor Optimization (No Training Required)")
        print("=" * 60)
        print(f"Image size: {self.image_size}")
        print(f"FPN strides: {self.strides}")
        print(f"Evaluating {self.num_samples} images, {self.total_gt_boxes} GT boxes")
        print(f"Running {n_trials} trials...")
        print("=" * 60)

        # First, evaluate default anchors as baseline
        default_sizes = tuple((s*2, s*4) for s in self.strides)
        default_ratios = (self.base_aspect_ratios,) * self.num_fpn_levels
        default_recalls = self.compute_geometric_recall(default_sizes, default_ratios)

        print(f"\nBaseline (stride-based default) anchors:")
        print(f"  Sizes: {default_sizes}")
        print(f"  Recalls: {default_recalls}")

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Extract best configuration
        best_trial = study.best_trial

        print("\n" + "=" * 60)
        print("Optimization Complete!")
        print("=" * 60)
        print(f"Best trial: {best_trial.number}")
        print(f"Best geometric recall score: {best_trial.value:.4f}")

        anchor_sizes = tuple(tuple(s) for s in best_trial.user_attrs["anchor_sizes"])
        aspect_ratios = best_trial.user_attrs["aspect_ratios"]
        recalls = best_trial.user_attrs["recalls"]

        print(f"\nBest anchor configuration:")
        print(f"  Sizes: {anchor_sizes}")
        print(f"  Aspect ratios: {aspect_ratios}")
        print(f"  Recalls: {recalls}")

        # Compare with baseline
        print(f"\nImprovement over baseline:")
        for k in recalls:
            improvement = (recalls[k] - default_recalls[k]) / max(default_recalls[k], 1e-6) * 100
            print(f"  {k}: {default_recalls[k]:.4f} -> {recalls[k]:.4f} ({improvement:+.1f}%)")

        return AnchorConfig(sizes=anchor_sizes, aspect_ratios=aspect_ratios)


# Alias for backward compatibility
AnchorOptimizer = GeometricAnchorOptimizer


def compare_anchor_configs(
    dataset,
    configs: Dict[str, AnchorConfig],
    image_size: Tuple[int, int] = (800, 800),
    num_samples: int = 500,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple anchor configurations on a dataset.

    Args:
        dataset: Dataset with targets
        configs: Dictionary of {name: AnchorConfig}
        image_size: Image size for anchor generation
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary of {name: {recall@0.5: ..., recall@0.7: ...}}
    """
    optimizer = GeometricAnchorOptimizer(
        dataset=dataset,
        image_size=image_size,
        num_samples=num_samples,
    )

    results = {}
    for name, config in configs.items():
        print(f"\nEvaluating: {name}")
        print(f"  Sizes: {config.sizes}")
        print(f"  Ratios: {config.aspect_ratios}")

        recalls = optimizer.compute_geometric_recall(
            config.sizes, config.aspect_ratios,
            iou_thresholds=[0.5, 0.7, 0.75]
        )
        results[name] = recalls

        print(f"  Recalls: {recalls}")

    return results

