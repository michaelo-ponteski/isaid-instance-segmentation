"""
Visualization module for Custom Mask R-CNN inference analysis.

Provides comprehensive visualization of intermediate stages:
- Grad-CAM heatmaps
- Backbone and FPN feature maps
- CBAM attention visualization
- RoI Align grid visualization
- Box and Mask head analysis
"""

from .gradcam_pipeline import (
    MaskRCNNVisualizationPipeline,
    GradCAMConfig,
    FeatureExtractor,
    GradCAM,
    load_pipeline,
    denormalize_image,
    overlay_heatmap,
    visualize_feature_maps,
    visualize_fpn_features,
    visualize_rpn_proposals,
    visualize_roi_align_grid,
    visualize_box_head_features,
    visualize_mask_head_stages,
    visualize_final_predictions,
    ISAID_CLASS_LABELS,
    ISAID_COLORS,
)

__all__ = [
    'MaskRCNNVisualizationPipeline',
    'GradCAMConfig',
    'FeatureExtractor',
    'GradCAM',
    'load_pipeline',
    'denormalize_image',
    'overlay_heatmap',
    'visualize_feature_maps',
    'visualize_fpn_features',
    'visualize_rpn_proposals',
    'visualize_roi_align_grid',
    'visualize_box_head_features',
    'visualize_mask_head_stages',
    'visualize_final_predictions',
    'ISAID_CLASS_LABELS',
    'ISAID_COLORS',
]
