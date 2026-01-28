"""
Custom Mask R-CNN model with modified architecture.
custom layers + attention
"""

import torch
import torch.nn as nn
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign

from models.backbone import BackboneWithFPN, build_custom_backbone_with_fpn
from models.roi_heads import (
    CustomMaskHead,
    CustomBoxFeatureExtractor,
    CustomBoxPredictor,
)


class CustomMaskRCNN(nn.Module):
    """
    Custom Mask R-CNN with:
    - EfficientNet backbone with CBAM attention
    - Custom FPN with attention modules
    - Enhanced RoI heads with additional layers

    """

    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        # Custom backbone support
        backbone_with_fpn=None,  # Optional: pass a pre-built BackboneWithFPN
        fpn_out_channels=256,    # FPN output channels (must match backbone if provided)
        # RPN parameters - anchors found with anchor optimizer
        rpn_anchor_sizes=((8, 16), (16, 32), (32, 64), (64, 128)),
        rpn_aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        # RoI parameters
        box_roi_pool_output_size=7,
        mask_roi_pool_output_size=14,
        box_head_hidden_dim=1024,
        mask_head_hidden_dim=256,
        # Inference parameters
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Backbone with FPN
        if backbone_with_fpn is not None:
            self.backbone = backbone_with_fpn
            fpn_out_channels = backbone_with_fpn.out_channels
        else:
            backbone, fpn, fpn_out_channels = build_custom_backbone_with_fpn(
                pretrained=pretrained_backbone
            )
            self.backbone = BackboneWithFPN(backbone, fpn)

        # RPN
        anchor_generator = AnchorGenerator(
            sizes=rpn_anchor_sizes, aspect_ratios=rpn_aspect_ratios
        )

        rpn_head = RPNHead(
            in_channels=fpn_out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        # RPN settings
        rpn_pre_nms_top_n = {"training": 2000, "testing": 1000}
        rpn_post_nms_top_n = {"training": 2000, "testing": 1000}

        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.5, # Foreground IoU threshold changed from 0.7 for better recall
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            nms_thresh=0.7,
        )

        # ROI Heads

        # Box RoI pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["P2", "P3", "P4", "P5"],
            output_size=box_roi_pool_output_size,
            sampling_ratio=2,
        )

        # Mask RoI pooling
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["P2", "P3", "P4", "P5"],
            output_size=mask_roi_pool_output_size,
            sampling_ratio=2,
        )

        # Custom box feature extractor
        box_head = CustomBoxFeatureExtractor(
            in_channels=fpn_out_channels * box_roi_pool_output_size**2,
            representation_size=box_head_hidden_dim,
        )

        # Custom box predictor
        box_predictor = CustomBoxPredictor(
            in_channels=box_head_hidden_dim,
            num_classes=num_classes,
        )

        # Custom mask head
        mask_head = CustomMaskHead(
            in_channels=fpn_out_channels,
            hidden_dim=mask_head_hidden_dim,
            num_classes=num_classes,
        )

        # RoI heads
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=None,  # Included in CustomMaskHead
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img,
        )

        # CustomMaskHead already outputs final mask logits
        self.roi_heads.mask_predictor = lambda x: x

    def forward(self, images, targets=None):
        """
        Args:
            images: List of tensors [C, H, W]
            targets: List of dicts with boxes, labels, masks

        Returns:
            Training or eval with targets: dict with losses
            Inference (eval without targets): list of dicts with predictions
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Handle both list of tensors and batched tensor input
        if isinstance(images, torch.Tensor):
            # If batched tensor [B, C, H, W], convert to list
            images = list(images)

        original_image_sizes = [img.shape[-2:] for img in images]

        # Stack images and create ImageList (required by RPN and RoIHeads)
        images_tensor = torch.stack(images)
        image_list = ImageList(images_tensor, original_image_sizes)

        # 1. Backbone forward pass
        features = self.backbone(images_tensor)

        # 2. RPN forward pass (expects ImageList, features dict, targets)
        proposals, proposal_losses = self.rpn(image_list, features, targets)

        # 3. RoI heads forward pass
        detections, detector_losses = self.roi_heads(
            features, proposals, image_list.image_sizes, targets
        )

        # 4. Aggregate losses or return predictions
        # Return losses if in training mode OR if targets are provided in eval mode
        if self.training or targets is not None:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        else:
            return detections

    def get_model_info(self):
        """Return model information for report"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calculate model size in MB
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": size_mb,
            "num_classes": self.num_classes,
        }


def get_custom_maskrcnn(
    num_classes,
    pretrained_backbone=True,
    rpn_anchor_sizes=None,
    rpn_aspect_ratios=None,
    optimized_anchors=False,
    anchor_config=None,
):
    """
    Factory function to create custom Mask R-CNN model.
    
    Args:
        num_classes: Number of classes including background
        pretrained_backbone: Whether to use pretrained backbone
        rpn_anchor_sizes: Custom anchor sizes (tuple of tuples)
        rpn_aspect_ratios: Custom aspect ratios (tuple of tuples)
        optimized_anchors: If True and anchor_config is None, use data-suggested anchors
        anchor_config: AnchorConfig object from anchor optimizer
        
    Returns:
        CustomMaskRCNN model
    """
    # Default anchor configuration for satellite imagery
    default_anchor_sizes = ((16, 24), (32, 48), (64, 96), (128, 192))
    default_aspect_ratios = ((0.5, 1.0, 2.0),) * 4
    
    # Use anchor_config if provided
    if anchor_config is not None:
        rpn_anchor_sizes = anchor_config.sizes
        rpn_aspect_ratios = anchor_config.aspect_ratios
    elif rpn_anchor_sizes is None:
        rpn_anchor_sizes = default_anchor_sizes
    
    if rpn_aspect_ratios is None:
        rpn_aspect_ratios = default_aspect_ratios
    
    return CustomMaskRCNN(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        rpn_anchor_sizes=rpn_anchor_sizes,
        rpn_aspect_ratios=rpn_aspect_ratios,
    )



