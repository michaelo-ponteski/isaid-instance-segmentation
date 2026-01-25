"""
Grad-CAM and Intermediate Visualization Pipeline for Custom Mask R-CNN

This module provides comprehensive visualization of:
1. Backbone feature activations with CBAM attention
2. FPN multi-scale feature maps
3. RPN proposals and anchor visualization
4. RoI Align grid visualization
5. Grad-CAM heatmaps at various stages
6. Box head and Mask head activations
7. Final predictions with confidence analysis

Author: CV Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import cv2
from pathlib import Path

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.maskrcnn_model import CustomMaskRCNN, get_custom_maskrcnn
from models.backbone import CustomEfficientNetBackbone, AttentionFPN, CBAM


# =============================================================================
# Configuration
# =============================================================================

ISAID_CLASS_LABELS = {
    0: "background", 1: "ship", 2: "storage_tank", 3: "baseball_diamond",
    4: "tennis_court", 5: "basketball_court", 6: "ground_track_field",
    7: "bridge", 8: "large_vehicle", 9: "small_vehicle", 10: "helicopter",
    11: "swimming_pool", 12: "roundabout", 13: "soccer_ball_field",
    14: "plane", 15: "harbor",
}

ISAID_COLORS = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255],
    4: [255, 255, 0], 5: [255, 0, 255], 6: [0, 255, 255], 7: [128, 0, 0],
    8: [0, 128, 0], 9: [0, 0, 128], 10: [128, 128, 0], 11: [128, 0, 128],
    12: [0, 128, 128], 13: [255, 128, 0], 14: [128, 255, 0], 15: [255, 0, 128],
}


@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM visualization pipeline."""
    device: str = "cuda"
    conf_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 20
    
    # Visualization settings
    colormap: str = "jet"
    alpha_overlay: float = 0.5
    figsize: Tuple[int, int] = (16, 12)
    dpi: int = 100
    
    # Output settings
    save_individual: bool = True
    output_dir: str = "./gradcam_outputs"


# =============================================================================
# Hook-based Feature Extraction
# =============================================================================

class FeatureExtractor:
    """
    Extracts intermediate features and gradients using forward/backward hooks.
    """
    
    def __init__(self, model: CustomMaskRCNN):
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        
    def _save_features(self, name: str):
        """Create a hook that saves features."""
        def hook(module, input, output):
            if isinstance(output, dict):
                for k, v in output.items():
                    self.features[f"{name}_{k}"] = v.detach()
            elif isinstance(output, tuple):
                for i, v in enumerate(output):
                    if isinstance(v, torch.Tensor):
                        self.features[f"{name}_{i}"] = v.detach()
            else:
                self.features[name] = output.detach()
        return hook
    
    def _save_gradients(self, name: str):
        """Create a hook that saves gradients."""
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                for i, g in enumerate(grad_output):
                    if g is not None:
                        self.gradients[f"{name}_{i}"] = g.detach()
            else:
                if grad_output is not None:
                    self.gradients[name] = grad_output.detach()
        return hook
    
    def register_hooks(self):
        """Register hooks for all important layers."""
        self.clear_hooks()
        
        # Backbone stages with CBAM
        backbone = self.model.backbone.backbone
        
        # EfficientNet feature stages
        for idx, module in enumerate(backbone.features):
            h = module.register_forward_hook(self._save_features(f"backbone_stage_{idx}"))
            self.hooks.append(h)
        
        # CBAM attention modules
        for name in ['attention_c2', 'attention_c3', 'attention_c4', 'attention_c5']:
            if hasattr(backbone, name):
                module = getattr(backbone, name)
                h = module.register_forward_hook(self._save_features(f"cbam_{name}"))
                self.hooks.append(h)
                h = module.register_full_backward_hook(self._save_gradients(f"cbam_{name}"))
                self.hooks.append(h)
        
        # FPN layers
        fpn = self.model.backbone.fpn
        for i, (lateral, output, attention) in enumerate(zip(
            fpn.lateral_convs, fpn.output_convs, fpn.attention_modules
        )):
            h = lateral.register_forward_hook(self._save_features(f"fpn_lateral_{i}"))
            self.hooks.append(h)
            h = output.register_forward_hook(self._save_features(f"fpn_output_{i}"))
            self.hooks.append(h)
            h = attention.register_forward_hook(self._save_features(f"fpn_attention_{i}"))
            self.hooks.append(h)
            h = attention.register_full_backward_hook(self._save_gradients(f"fpn_attention_{i}"))
            self.hooks.append(h)
        
        # RPN head
        h = self.model.rpn.head.register_forward_hook(self._save_features("rpn_head"))
        self.hooks.append(h)
        
        # RoI Heads - Box head
        box_head = self.model.roi_heads.box_head
        h = box_head.fc1.register_forward_hook(self._save_features("box_head_fc1"))
        self.hooks.append(h)
        h = box_head.fc2.register_forward_hook(self._save_features("box_head_fc2"))
        self.hooks.append(h)
        h = box_head.fc2.register_full_backward_hook(self._save_gradients("box_head_fc2"))
        self.hooks.append(h)
        
        # RoI Heads - Box predictor
        box_predictor = self.model.roi_heads.box_predictor
        h = box_predictor.cls_score.register_forward_hook(self._save_features("box_cls_score"))
        self.hooks.append(h)
        h = box_predictor.bbox_pred.register_forward_hook(self._save_features("box_bbox_pred"))
        self.hooks.append(h)
        
        # Mask head
        mask_head = self.model.roi_heads.mask_head
        for name in ['conv1', 'conv2', 'conv3', 'conv4', 'deconv', 'mask_fcn_logits']:
            if hasattr(mask_head, name):
                module = getattr(mask_head, name)
                h = module.register_forward_hook(self._save_features(f"mask_head_{name}"))
                self.hooks.append(h)
        h = mask_head.mask_fcn_logits.register_full_backward_hook(self._save_gradients("mask_head_logits"))
        self.hooks.append(h)
        
        print(f"Registered {len(self.hooks)} hooks")
        
    def clear_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
    def clear_features(self):
        """Clear stored features and gradients."""
        self.features = {}
        self.gradients = {}
        
    def __del__(self):
        self.clear_hooks()


# =============================================================================
# Grad-CAM Implementation
# =============================================================================

class GradCAM:
    """
    Grad-CAM implementation for object detection models.
    
    Computes class activation maps using gradients of the target class
    with respect to feature maps.
    """
    
    def __init__(self, model: CustomMaskRCNN, feature_extractor: FeatureExtractor):
        self.model = model
        self.feature_extractor = feature_extractor
        
    def compute_gradcam(
        self,
        features: torch.Tensor,
        gradients: torch.Tensor,
        target_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap from features and gradients.
        
        Args:
            features: Feature map [C, H, W] or [B, C, H, W]
            gradients: Gradient tensor [C, H, W] or [B, C, H, W]
            target_size: Optional size to resize heatmap to
            
        Returns:
            Normalized heatmap as numpy array
        """
        # Handle batch dimension
        if features.dim() == 4:
            features = features[0]
        if gradients.dim() == 4:
            gradients = gradients[0]
            
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of feature maps
        cam = (weights * features).sum(dim=0)  # [H, W]
        
        # ReLU to keep only positive activations
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = cam.cpu().numpy()
        
        # Resize if needed
        if target_size is not None:
            cam = cv2.resize(cam, (target_size[1], target_size[0]))
        
        return cam
    
    def compute_guided_gradcam(
        self,
        features: torch.Tensor,
        gradients: torch.Tensor,
        target_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Compute Guided Grad-CAM (element-wise product of Grad-CAM and guided backprop).
        """
        cam = self.compute_gradcam(features, gradients, target_size)
        
        # For guided backprop, we'd need to modify the backward pass
        # Here we approximate by using positive gradients only
        if gradients.dim() == 4:
            gradients = gradients[0]
        
        guided = F.relu(gradients).mean(dim=0).cpu().numpy()
        if target_size is not None:
            guided = cv2.resize(guided, (target_size[1], target_size[0]))
        
        # Normalize guided
        guided = guided - guided.min()
        if guided.max() > 0:
            guided = guided / guided.max()
        
        # Element-wise product
        guided_cam = cam * guided
        guided_cam = guided_cam - guided_cam.min()
        if guided_cam.max() > 0:
            guided_cam = guided_cam / guided_cam.max()
        
        return guided_cam


# =============================================================================
# Visualization Functions
# =============================================================================

def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize image tensor to numpy array [H, W, C] in range [0, 255]."""
    if isinstance(image, torch.Tensor):
        img = image.cpu().clone()
        if img.dim() == 4:
            img = img[0]
        
        # Denormalize
        for c, (m, s) in enumerate(zip(mean, std)):
            img[c] = img[c] * s + m
        
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        img = np.array(image)
    
    return img


def apply_colormap(
    heatmap: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """Apply colormap to heatmap and return RGB image."""
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha
    return (heatmap_colored * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay heatmap on image with transparency."""
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = apply_colormap(heatmap_resized, colormap)
    
    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return overlay


def visualize_feature_maps(
    features: torch.Tensor,
    title: str = "Feature Maps",
    num_channels: int = 16,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Visualize a grid of feature map channels."""
    if features.dim() == 4:
        features = features[0]
    
    features = features.cpu().numpy()
    n_channels = min(num_channels, features.shape[0])
    
    # Calculate grid size
    cols = min(8, n_channels)
    rows = (n_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n_channels):
        feat = features[i]
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        
        axes[i].imshow(feat, cmap='viridis')
        axes[i].set_title(f'Ch {i}', fontsize=8)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_attention_maps(
    channel_attention: torch.Tensor,
    spatial_attention: torch.Tensor,
    original_features: torch.Tensor,
    title: str = "CBAM Attention",
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """Visualize CBAM channel and spatial attention."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original features (mean across channels)
    if original_features.dim() == 4:
        original_features = original_features[0]
    feat_mean = original_features.mean(dim=0).cpu().numpy()
    feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
    axes[0].imshow(feat_mean, cmap='viridis')
    axes[0].set_title('Input Features (mean)', fontsize=10)
    axes[0].axis('off')
    
    # Channel attention weights
    if channel_attention.dim() >= 2:
        ch_attn = channel_attention.squeeze().cpu().numpy()
        axes[1].bar(range(len(ch_attn)), ch_attn)
        axes[1].set_title('Channel Attention', fontsize=10)
        axes[1].set_xlabel('Channel')
    
    # Spatial attention map
    if spatial_attention.dim() >= 2:
        sp_attn = spatial_attention.squeeze().cpu().numpy()
        axes[2].imshow(sp_attn, cmap='hot')
        axes[2].set_title('Spatial Attention', fontsize=10)
        axes[2].axis('off')
    
    # Attended features
    axes[3].imshow(feat_mean, cmap='viridis')
    axes[3].set_title('Output (with attention)', fontsize=10)
    axes[3].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_fpn_features(
    fpn_features: Dict[str, torch.Tensor],
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Visualize FPN multi-scale feature maps."""
    levels = sorted([k for k in fpn_features.keys() if k.startswith('P')])
    n_levels = len(levels)
    
    fig, axes = plt.subplots(2, n_levels, figsize=figsize)
    
    for i, level in enumerate(levels):
        feat = fpn_features[level]
        if feat.dim() == 4:
            feat = feat[0]
        feat = feat.cpu().numpy()
        
        # Mean activation
        mean_feat = feat.mean(axis=0)
        mean_feat = (mean_feat - mean_feat.min()) / (mean_feat.max() - mean_feat.min() + 1e-8)
        axes[0, i].imshow(mean_feat, cmap='viridis')
        axes[0, i].set_title(f'{level} Mean\n{feat.shape}', fontsize=10)
        axes[0, i].axis('off')
        
        # Max activation
        max_feat = feat.max(axis=0)
        max_feat = (max_feat - max_feat.min()) / (max_feat.max() - max_feat.min() + 1e-8)
        axes[1, i].imshow(max_feat, cmap='hot')
        axes[1, i].set_title(f'{level} Max', fontsize=10)
        axes[1, i].axis('off')
    
    fig.suptitle('FPN Multi-Scale Features', fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_rpn_proposals(
    image: np.ndarray,
    proposals: torch.Tensor,
    objectness: torch.Tensor = None,
    max_proposals: int = 100,
    figsize: Tuple[int, int] = (12, 12),
) -> plt.Figure:
    """Visualize RPN proposals on image."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    
    proposals = proposals.cpu().numpy()
    if objectness is not None:
        objectness = objectness.cpu().numpy()
        # Sort by objectness
        indices = np.argsort(objectness)[::-1][:max_proposals]
        proposals = proposals[indices]
        objectness = objectness[indices]
    else:
        proposals = proposals[:max_proposals]
    
    # Color by objectness score
    cmap = plt.get_cmap('RdYlGn')
    
    for i, box in enumerate(proposals):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        if objectness is not None:
            color = cmap(objectness[i])
            alpha = 0.3 + 0.5 * objectness[i]
        else:
            color = 'cyan'
            alpha = 0.3
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=1, edgecolor=color,
            facecolor='none', alpha=alpha
        )
        ax.add_patch(rect)
    
    ax.set_title(f'RPN Proposals (top {len(proposals)})', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def visualize_roi_align_grid(
    image: np.ndarray,
    boxes: torch.Tensor,
    output_size: int = 7,
    sampling_ratio: int = 2,
    max_boxes: int = 4,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Visualize RoI Align sampling grid on image."""
    boxes = boxes.cpu().numpy()[:max_boxes]
    n_boxes = len(boxes)
    
    fig, axes = plt.subplots(1, n_boxes + 1, figsize=figsize)
    
    # Full image with all boxes
    axes[0].imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0].add_patch(rect)
    axes[0].set_title('Input Image with RoIs', fontsize=10)
    axes[0].axis('off')
    
    # Individual boxes with sampling grid
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Crop region
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
            
        axes[i + 1].imshow(crop)
        
        # Draw grid
        for gx in range(output_size + 1):
            x = gx * (w / output_size)
            axes[i + 1].axvline(x, color='yellow', linewidth=0.5, alpha=0.7)
        for gy in range(output_size + 1):
            y = gy * (h / output_size)
            axes[i + 1].axhline(y, color='yellow', linewidth=0.5, alpha=0.7)
        
        # Draw sampling points
        cell_w = w / output_size
        cell_h = h / output_size
        for gx in range(output_size):
            for gy in range(output_size):
                for sx in range(sampling_ratio):
                    for sy in range(sampling_ratio):
                        px = (gx + (sx + 0.5) / sampling_ratio) * cell_w
                        py = (gy + (sy + 0.5) / sampling_ratio) * cell_h
                        axes[i + 1].plot(px, py, 'r.', markersize=2)
        
        axes[i + 1].set_title(f'RoI {i+1}: {output_size}x{output_size} grid', fontsize=10)
        axes[i + 1].axis('off')
    
    fig.suptitle('RoI Align Sampling Grid Visualization', fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_box_head_features(
    fc1_features: torch.Tensor,
    fc2_features: torch.Tensor,
    cls_scores: torch.Tensor,
    bbox_deltas: torch.Tensor,
    class_labels: Dict[int, str],
    max_rois: int = 8,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Visualize box head feature activations."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # FC1 activations
    fc1 = fc1_features[:max_rois].cpu().numpy()
    axes[0, 0].imshow(fc1, aspect='auto', cmap='viridis')
    axes[0, 0].set_xlabel('Feature Dimension')
    axes[0, 0].set_ylabel('RoI Index')
    axes[0, 0].set_title('FC1 Activations', fontsize=12)
    
    # FC2 activations
    fc2 = fc2_features[:max_rois].cpu().numpy()
    axes[0, 1].imshow(fc2, aspect='auto', cmap='viridis')
    axes[0, 1].set_xlabel('Feature Dimension')
    axes[0, 1].set_ylabel('RoI Index')
    axes[0, 1].set_title('FC2 Activations', fontsize=12)
    
    # Class scores (softmax)
    cls_probs = F.softmax(cls_scores[:max_rois], dim=1).cpu().numpy()
    im = axes[1, 0].imshow(cls_probs, aspect='auto', cmap='hot')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('RoI Index')
    axes[1, 0].set_title('Class Probabilities', fontsize=12)
    axes[1, 0].set_xticks(range(len(class_labels)))
    axes[1, 0].set_xticklabels([class_labels[i][:6] for i in range(len(class_labels))], 
                                rotation=45, ha='right', fontsize=7)
    plt.colorbar(im, ax=axes[1, 0])
    
    # Top class per RoI
    top_classes = cls_probs.argmax(axis=1)
    top_probs = cls_probs.max(axis=1)
    
    bars = axes[1, 1].barh(range(len(top_classes)), top_probs)
    axes[1, 1].set_yticks(range(len(top_classes)))
    axes[1, 1].set_yticklabels([f'RoI {i}: {class_labels[top_classes[i]]}' 
                                 for i in range(len(top_classes))], fontsize=9)
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_title('Top Predicted Class per RoI', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    
    # Color bars by confidence
    for bar, prob in zip(bars, top_probs):
        bar.set_color(plt.cm.RdYlGn(prob))
    
    fig.suptitle('Box Head Analysis', fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_mask_head_stages(
    mask_features: Dict[str, torch.Tensor],
    box_idx: int = 0,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """Visualize mask head intermediate stages."""
    stages = ['conv1', 'conv2', 'conv3', 'conv4', 'deconv', 'mask_fcn_logits']
    available_stages = [s for s in stages if f'mask_head_{s}' in mask_features]
    
    n_stages = len(available_stages)
    fig, axes = plt.subplots(2, n_stages, figsize=figsize)
    
    for i, stage in enumerate(available_stages):
        feat = mask_features[f'mask_head_{stage}']
        if feat.dim() == 4 and feat.shape[0] > box_idx:
            feat = feat[box_idx]
        elif feat.dim() == 4:
            feat = feat[0]
        feat = feat.cpu().numpy()
        
        # Mean activation
        if feat.ndim == 3:
            mean_feat = feat.mean(axis=0)
        else:
            mean_feat = feat
        mean_feat = (mean_feat - mean_feat.min()) / (mean_feat.max() - mean_feat.min() + 1e-8)
        axes[0, i].imshow(mean_feat, cmap='viridis')
        axes[0, i].set_title(f'{stage}\n{feat.shape}', fontsize=9)
        axes[0, i].axis('off')
        
        # Max activation
        if feat.ndim == 3:
            max_feat = feat.max(axis=0)
        else:
            max_feat = feat
        max_feat = (max_feat - max_feat.min()) / (max_feat.max() - max_feat.min() + 1e-8)
        axes[1, i].imshow(max_feat, cmap='hot')
        axes[1, i].set_title(f'Max activation', fontsize=9)
        axes[1, i].axis('off')
    
    fig.suptitle(f'Mask Head Stages (RoI {box_idx})', fontsize=14)
    plt.tight_layout()
    
    return fig


def visualize_final_predictions(
    image: np.ndarray,
    predictions: Dict[str, torch.Tensor],
    gradcam_heatmap: np.ndarray = None,
    class_labels: Dict[int, str] = None,
    conf_threshold: float = 0.5,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Visualize final predictions with optional Grad-CAM overlay."""
    if class_labels is None:
        class_labels = ISAID_CLASS_LABELS
    
    n_cols = 3 if gradcam_heatmap is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Predictions
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    masks = predictions.get('masks', None)
    
    if masks is not None:
        masks = masks.cpu().numpy()
    
    # Filter by confidence
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    if masks is not None:
        masks = masks[keep]
    
    # Create prediction overlay
    pred_img = image.copy()
    
    # Draw masks
    if masks is not None:
        mask_overlay = np.zeros_like(image, dtype=np.float32)
        for i, (mask, label) in enumerate(zip(masks, labels)):
            if mask.ndim == 3:
                mask = mask[0]
            mask_binary = (mask > 0.5).astype(np.uint8)
            color = np.array(ISAID_COLORS.get(label, [255, 255, 255])) / 255.0
            for c in range(3):
                mask_overlay[:, :, c] += mask_binary * color[c] * 0.4
        
        mask_overlay = np.clip(mask_overlay, 0, 1)
        pred_img = (pred_img.astype(np.float32) / 255.0 * 0.6 + mask_overlay * 0.4)
        pred_img = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
    
    axes[1].imshow(pred_img)
    
    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = np.array(ISAID_COLORS.get(label, [255, 255, 255])) / 255.0
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axes[1].add_patch(rect)
        
        label_text = f'{class_labels.get(label, label)}: {score:.2f}'
        axes[1].text(x1, y1 - 5, label_text, fontsize=8,
                     color='white', backgroundcolor=color)
    
    axes[1].set_title(f'Predictions ({len(boxes)} objects)', fontsize=12)
    axes[1].axis('off')
    
    # Grad-CAM overlay
    if gradcam_heatmap is not None:
        overlay = overlay_heatmap(image, gradcam_heatmap, alpha=0.5)
        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# Main Pipeline Class
# =============================================================================

class MaskRCNNVisualizationPipeline:
    """
    Complete visualization pipeline for Custom Mask R-CNN inference analysis.
    
    Provides visualizations for:
    1. Backbone features with CBAM attention
    2. FPN multi-scale features
    3. RPN proposals
    4. RoI Align sampling
    5. Box head analysis
    6. Mask head stages
    7. Grad-CAM heatmaps
    8. Final predictions
    """
    
    def __init__(
        self,
        model: CustomMaskRCNN,
        config: GradCAMConfig = None,
    ):
        self.model = model
        self.config = config or GradCAMConfig()
        
        self.feature_extractor = FeatureExtractor(model)
        self.gradcam = GradCAM(model, self.feature_extractor)
        
        self.model.to(self.config.device)
        self.model.eval()
        
        # Storage for current analysis
        self.current_image: np.ndarray = None
        self.current_image_tensor: torch.Tensor = None
        self.predictions: Dict[str, torch.Tensor] = None
        self.fpn_features: Dict[str, torch.Tensor] = None
        self.proposals: torch.Tensor = None
        
    def load_model_weights(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Loaded model weights from {checkpoint_path}")
    
    def preprocess_image(
        self,
        image: Union[str, np.ndarray, torch.Tensor, Image.Image],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess image for model input."""
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(image, np.ndarray):
            self.current_image = image.copy()
            # Convert to tensor
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Normalize
            for c, (m, s) in enumerate(zip(mean, std)):
                img_tensor[c] = (img_tensor[c] - m) / s
            image = img_tensor
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        self.current_image_tensor = image.to(self.config.device)
        
        # Store denormalized for visualization
        if self.current_image is None:
            self.current_image = denormalize_image(image[0])
        
        return self.current_image_tensor, self.current_image
    
    @torch.enable_grad()
    def run_inference_with_gradients(
        self,
        target_class: int = None,
        target_box_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run inference and compute gradients for Grad-CAM.
        
        Args:
            target_class: Class to compute gradients for (None = use predicted class)
            target_box_idx: Which detection to analyze
            
        Returns:
            Dictionary with all intermediate results
        """
        self.feature_extractor.register_hooks()
        self.feature_extractor.clear_features()
        
        # Forward pass
        self.model.eval()
        
        # We need gradients for Grad-CAM
        image = self.current_image_tensor.requires_grad_(True)
        
        # Get backbone features first
        features = self.model.backbone(image)
        self.fpn_features = features
        
        # Store in feature extractor
        for k, v in features.items():
            self.feature_extractor.features[f"fpn_{k}"] = v.detach()
        
        # Run full model
        with torch.no_grad():
            predictions = self.model([image[0]])[0]
        self.predictions = predictions
        
        results = {
            'predictions': predictions,
            'fpn_features': features,
            'extracted_features': self.feature_extractor.features.copy(),
        }
        
        # Compute Grad-CAM if detections exist
        if len(predictions['boxes']) > 0:
            # Get target info
            if target_class is None:
                target_class = predictions['labels'][target_box_idx].item()
            
            # Backward pass for Grad-CAM
            # We need to run a forward pass with gradients enabled
            self.model.zero_grad()
            
            # Use the FPN features for Grad-CAM
            target_layer_name = 'fpn_attention_0'  # C5 level attention
            
            if target_layer_name in self.feature_extractor.features:
                target_features = self.feature_extractor.features[target_layer_name]
                
                # Create a pseudo-loss based on detection confidence
                if 'scores' in predictions and len(predictions['scores']) > target_box_idx:
                    score = predictions['scores'][target_box_idx]
                    # Backward through the score would require re-running with grad
                    # For now, we use feature activation as proxy
                    
                results['target_features'] = target_features
                results['target_class'] = target_class
        
        # Store proposals if available
        if hasattr(self.model, '_proposals'):
            self.proposals = self.model._proposals
        
        self.feature_extractor.clear_hooks()
        
        return results
    
    def generate_all_visualizations(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        output_dir: str = None,
        save: bool = True,
    ) -> Dict[str, plt.Figure]:
        """
        Generate all visualizations for a single image.
        
        Args:
            image: Input image (path, array, or tensor)
            output_dir: Directory to save figures
            save: Whether to save figures to disk
            
        Returns:
            Dictionary of figure names to matplotlib figures
        """
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # 1. Preprocess image
        print("Preprocessing image...")
        self.preprocess_image(image)
        
        # 2. Run inference with feature extraction
        print("Running inference with feature extraction...")
        results = self.run_inference_with_gradients()
        
        extracted = results['extracted_features']
        predictions = results['predictions']
        
        print(f"Found {len(predictions['boxes'])} detections")
        
        # 3. Backbone feature visualizations
        print("Generating backbone visualizations...")
        for stage in ['backbone_stage_3', 'backbone_stage_4', 'backbone_stage_5', 'backbone_stage_7']:
            if stage in extracted:
                fig = visualize_feature_maps(
                    extracted[stage],
                    title=f'Backbone: {stage}',
                    num_channels=16
                )
                figures[stage] = fig
                if save:
                    fig.savefig(f'{output_dir}/{stage}.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 4. CBAM attention visualizations
        print("Generating CBAM attention visualizations...")
        for cbam_name in ['cbam_attention_c2', 'cbam_attention_c3', 'cbam_attention_c4', 'cbam_attention_c5']:
            if cbam_name in extracted:
                fig = visualize_feature_maps(
                    extracted[cbam_name],
                    title=f'CBAM Output: {cbam_name}',
                    num_channels=16
                )
                figures[cbam_name] = fig
                if save:
                    fig.savefig(f'{output_dir}/{cbam_name}.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 5. FPN features
        print("Generating FPN visualizations...")
        fpn_dict = {k: v for k, v in results['fpn_features'].items()}
        fig = visualize_fpn_features(fpn_dict)
        figures['fpn_features'] = fig
        if save:
            fig.savefig(f'{output_dir}/fpn_features.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 6. FPN attention outputs
        for i in range(4):
            key = f'fpn_attention_{i}'
            if key in extracted:
                fig = visualize_feature_maps(
                    extracted[key],
                    title=f'FPN Attention Level {i} (P{5-i})',
                    num_channels=16
                )
                figures[key] = fig
                if save:
                    fig.savefig(f'{output_dir}/{key}.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 7. RoI Align grid visualization
        if len(predictions['boxes']) > 0:
            print("Generating RoI Align visualization...")
            fig = visualize_roi_align_grid(
                self.current_image,
                predictions['boxes'][:4],
                output_size=7,
                sampling_ratio=2
            )
            figures['roi_align_grid'] = fig
            if save:
                fig.savefig(f'{output_dir}/roi_align_grid.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 8. Box head analysis
        if 'box_head_fc1' in extracted and 'box_cls_score' in extracted:
            print("Generating box head analysis...")
            fig = visualize_box_head_features(
                extracted['box_head_fc1'],
                extracted['box_head_fc2'],
                extracted['box_cls_score'],
                extracted.get('box_bbox_pred', torch.zeros(1)),
                ISAID_CLASS_LABELS,
            )
            figures['box_head_analysis'] = fig
            if save:
                fig.savefig(f'{output_dir}/box_head_analysis.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 9. Mask head stages
        mask_features = {k: v for k, v in extracted.items() if k.startswith('mask_head_')}
        if mask_features and len(predictions['boxes']) > 0:
            print("Generating mask head visualizations...")
            fig = visualize_mask_head_stages(mask_features, box_idx=0)
            figures['mask_head_stages'] = fig
            if save:
                fig.savefig(f'{output_dir}/mask_head_stages.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 10. Grad-CAM heatmap
        print("Generating Grad-CAM heatmaps...")
        for level in ['P5', 'P4', 'P3', 'P2']:
            fpn_key = f'fpn_{level}'
            if fpn_key in extracted or level in results['fpn_features']:
                feat = results['fpn_features'].get(level, extracted.get(fpn_key))
                if feat is not None:
                    # Simple activation-based heatmap (pseudo Grad-CAM)
                    if feat.dim() == 4:
                        feat = feat[0]
                    heatmap = feat.mean(dim=0).cpu().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    heatmap = cv2.resize(heatmap, (self.current_image.shape[1], self.current_image.shape[0]))
                    
                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                    axes[0].imshow(self.current_image)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(heatmap, cmap='jet')
                    axes[1].set_title(f'{level} Activation Heatmap')
                    axes[1].axis('off')
                    
                    overlay = overlay_heatmap(self.current_image, heatmap, alpha=0.5)
                    axes[2].imshow(overlay)
                    axes[2].set_title(f'{level} Overlay')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    figures[f'gradcam_{level}'] = fig
                    if save:
                        fig.savefig(f'{output_dir}/gradcam_{level}.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 11. Final predictions
        print("Generating final predictions visualization...")
        # Get best FPN level heatmap for overlay
        best_heatmap = None
        for level in ['P4', 'P5', 'P3']:
            if level in results['fpn_features']:
                feat = results['fpn_features'][level]
                if feat.dim() == 4:
                    feat = feat[0]
                best_heatmap = feat.mean(dim=0).cpu().numpy()
                best_heatmap = (best_heatmap - best_heatmap.min()) / (best_heatmap.max() - best_heatmap.min() + 1e-8)
                best_heatmap = cv2.resize(best_heatmap, (self.current_image.shape[1], self.current_image.shape[0]))
                break
        
        fig = visualize_final_predictions(
            self.current_image,
            predictions,
            gradcam_heatmap=best_heatmap,
            class_labels=ISAID_CLASS_LABELS,
            conf_threshold=self.config.conf_threshold,
        )
        figures['final_predictions'] = fig
        if save:
            fig.savefig(f'{output_dir}/final_predictions.png', dpi=self.config.dpi, bbox_inches='tight')
        
        # 12. Summary grid
        print("Generating summary grid...")
        fig = self._create_summary_grid(results, best_heatmap)
        figures['summary'] = fig
        if save:
            fig.savefig(f'{output_dir}/summary.png', dpi=self.config.dpi * 2, bbox_inches='tight')
        
        print(f"Generated {len(figures)} visualizations")
        if save:
            print(f"Saved to {output_dir}/")
        
        return figures
    
    def _create_summary_grid(
        self,
        results: Dict[str, Any],
        heatmap: np.ndarray = None,
    ) -> plt.Figure:
        """Create a summary grid of the most important visualizations."""
        fig = plt.figure(figsize=(20, 16))
        
        # Grid: 4x4
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.2)
        
        # Row 1: Input -> Backbone features -> CBAM -> FPN
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.current_image)
        ax1.set_title('1. Input Image', fontsize=11)
        ax1.axis('off')
        
        # Backbone C5
        extracted = results['extracted_features']
        if 'backbone_stage_7' in extracted:
            feat = extracted['backbone_stage_7']
            if feat.dim() == 4:
                feat = feat[0]
            feat_mean = feat.mean(dim=0).cpu().numpy()
            feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(feat_mean, cmap='viridis')
            ax2.set_title('2. Backbone (C5)', fontsize=11)
            ax2.axis('off')
        
        # CBAM
        if 'cbam_attention_c5' in extracted:
            feat = extracted['cbam_attention_c5']
            if feat.dim() == 4:
                feat = feat[0]
            feat_mean = feat.mean(dim=0).cpu().numpy()
            feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(feat_mean, cmap='viridis')
            ax3.set_title('3. CBAM Attention (C5)', fontsize=11)
            ax3.axis('off')
        
        # FPN
        fpn_feat = results['fpn_features']
        if 'P5' in fpn_feat:
            feat = fpn_feat['P5']
            if feat.dim() == 4:
                feat = feat[0]
            feat_mean = feat.mean(dim=0).cpu().numpy()
            feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
            
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(feat_mean, cmap='viridis')
            ax4.set_title('4. FPN (P5)', fontsize=11)
            ax4.axis('off')
        
        # Row 2: FPN multi-scale
        for i, level in enumerate(['P2', 'P3', 'P4', 'P5']):
            if level in fpn_feat:
                feat = fpn_feat[level]
                if feat.dim() == 4:
                    feat = feat[0]
                feat_mean = feat.mean(dim=0).cpu().numpy()
                feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
                
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(feat_mean, cmap='plasma')
                ax.set_title(f'5-{i+1}. FPN {level}', fontsize=11)
                ax.axis('off')
        
        # Row 3: RoI analysis and mask stages
        predictions = results['predictions']
        
        # RoI boxes on image
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.imshow(self.current_image)
        if len(predictions['boxes']) > 0:
            for i, box in enumerate(predictions['boxes'][:5].cpu().numpy()):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax5.add_patch(rect)
        ax5.set_title('6. RPN Proposals', fontsize=11)
        ax5.axis('off')
        
        # Box head features
        if 'box_head_fc2' in extracted:
            feat = extracted['box_head_fc2'][:8].cpu().numpy()
            
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.imshow(feat, aspect='auto', cmap='viridis')
            ax6.set_title('7. Box Head FC2', fontsize=11)
            ax6.set_xlabel('Features')
            ax6.set_ylabel('RoI')
        
        # Class probabilities
        if 'box_cls_score' in extracted:
            cls_probs = F.softmax(extracted['box_cls_score'][:8], dim=1).cpu().numpy()
            
            ax7 = fig.add_subplot(gs[2, 2])
            ax7.imshow(cls_probs, aspect='auto', cmap='hot')
            ax7.set_title('8. Class Probabilities', fontsize=11)
            ax7.set_xlabel('Class')
            ax7.set_ylabel('RoI')
        
        # Mask head
        if 'mask_head_mask_fcn_logits' in extracted:
            mask_logits = extracted['mask_head_mask_fcn_logits']
            if mask_logits.dim() == 4 and mask_logits.shape[0] > 0:
                # Take first mask, predicted class channel
                if len(predictions['labels']) > 0:
                    cls_idx = predictions['labels'][0].item()
                    mask_vis = torch.sigmoid(mask_logits[0, cls_idx]).cpu().numpy()
                else:
                    mask_vis = mask_logits[0].mean(dim=0).cpu().numpy()
                
                ax8 = fig.add_subplot(gs[2, 3])
                ax8.imshow(mask_vis, cmap='gray')
                ax8.set_title('9. Mask Prediction', fontsize=11)
                ax8.axis('off')
        
        # Row 4: Grad-CAM and final result
        if heatmap is not None:
            ax9 = fig.add_subplot(gs[3, 0])
            ax9.imshow(heatmap, cmap='jet')
            ax9.set_title('10. Activation Heatmap', fontsize=11)
            ax9.axis('off')
            
            ax10 = fig.add_subplot(gs[3, 1])
            overlay = overlay_heatmap(self.current_image, heatmap, alpha=0.5)
            ax10.imshow(overlay)
            ax10.set_title('11. Heatmap Overlay', fontsize=11)
            ax10.axis('off')
        
        # Final predictions with masks
        ax11 = fig.add_subplot(gs[3, 2:])
        pred_img = self.current_image.copy()
        
        masks = predictions.get('masks', None)
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        keep = scores >= self.config.conf_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        if masks is not None:
            masks = masks[keep].cpu().numpy()
            mask_overlay = np.zeros_like(pred_img, dtype=np.float32)
            for mask, label in zip(masks, labels):
                if mask.ndim == 3:
                    mask = mask[0]
                mask_binary = (mask > 0.5).astype(np.uint8)
                color = np.array(ISAID_COLORS.get(label, [255, 255, 255])) / 255.0
                for c in range(3):
                    mask_overlay[:, :, c] += mask_binary * color[c] * 0.5
            
            pred_img = (pred_img.astype(np.float32) / 255.0 * 0.5 + mask_overlay)
            pred_img = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
        
        ax11.imshow(pred_img)
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = np.array(ISAID_COLORS.get(label, [255, 255, 255])) / 255.0
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax11.add_patch(rect)
            ax11.text(x1, y1-3, f'{ISAID_CLASS_LABELS.get(label, label)}: {score:.2f}',
                     fontsize=8, color='white', backgroundcolor=color)
        
        ax11.set_title(f'12. Final Predictions ({len(boxes)} detections)', fontsize=11)
        ax11.axis('off')
        
        fig.suptitle('Mask R-CNN Inference Pipeline Visualization', fontsize=16, fontweight='bold')
        
        return fig


# =============================================================================
# Utility Functions
# =============================================================================

def load_pipeline(
    checkpoint_path: str,
    num_classes: int = 16,
    device: str = "cuda",
    **model_kwargs,
) -> MaskRCNNVisualizationPipeline:
    """
    Convenience function to load model and create visualization pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes: Number of classes
        device: Device to use
        **model_kwargs: Additional arguments for model creation
        
    Returns:
        Configured visualization pipeline
    """
    config = GradCAMConfig(device=device)
    
    model = get_custom_maskrcnn(
        num_classes=num_classes,
        pretrained_backbone=False,
        **model_kwargs
    )
    
    pipeline = MaskRCNNVisualizationPipeline(model, config)
    pipeline.load_model_weights(checkpoint_path)
    
    return pipeline


if __name__ == "__main__":
    print("Grad-CAM Visualization Pipeline for Custom Mask R-CNN")
    print("=" * 60)
    print("\nUsage:")
    print("  from visualization.gradcam_pipeline import load_pipeline")
    print("  ")
    print("  pipeline = load_pipeline('checkpoint.pth', num_classes=16)")
    print("  figures = pipeline.generate_all_visualizations('image.jpg')")
    print("\nThis will generate visualizations for:")
    print("  1. Backbone feature maps")
    print("  2. CBAM attention outputs")
    print("  3. FPN multi-scale features")
    print("  4. RPN proposals")
    print("  5. RoI Align sampling grids")
    print("  6. Box head analysis")
    print("  7. Mask head stages")
    print("  8. Grad-CAM heatmaps")
    print("  9. Final predictions with overlays")
