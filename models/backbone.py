"""
Custom backbone with Attention for Mask R-CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ChannelAttention(nn.Module):
    """Channel Attention Module - own layer"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module - own layer"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module - own layer"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CustomEfficientNetBackbone(nn.Module):
    """
    Custom backbone based on EfficientNet with attention modules.
    Returns multi-scale features for FPN.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            efficientnet = efficientnet_b0(weights=weights)
        else:
            efficientnet = efficientnet_b0(weights=None)

        # Extract feature extraction layers
        self.features = efficientnet.features

        self.out_channels = {"C2": 40, "C3": 80, "C4": 112, "C5": 320}

        # Add attention modules to each stage (własne warstwy)
        self.attention_c2 = CBAM(40)
        self.attention_c3 = CBAM(80)
        self.attention_c4 = CBAM(112)
        self.attention_c5 = CBAM(320)

    def forward(self, x):
        features = {}

        # Pass through backbone and extract features at different scales
        out = x
        for idx, module in enumerate(self.features):
            out = module(out)

            if idx == 2:  # After stem, before first MBConv
                features["C1"] = out
            elif idx == 3:  # C2
                out = self.attention_c2(out)
                features["C2"] = out
            elif idx == 4:  # C3
                out = self.attention_c3(out)
                features["C3"] = out
            elif idx == 5:  # C4
                out = self.attention_c4(out)
                features["C4"] = out
            elif idx == 7:  # C5 - before final 1280ch expansion
                out = self.attention_c5(out)
                features["C5"] = out
                break  # Don't process final expansion layer

        return features


class AttentionFPN(nn.Module):
    """
    Custom Feature Pyramid Network with Attention
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list]
        )

        # Output convolutions (3x3 conv for smoothing)
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ]
        )

        # Attention modules for FPN outputs
        self.attention_modules = nn.ModuleList(
            [CBAM(out_channels) for _ in in_channels_list]
        )

        self.out_channels = out_channels

    def forward(self, features_dict):
        """
        Args:
            features_dict: Dictionary with keys ['C2', 'C3', 'C4', 'C5']
        Returns:
            Dictionary with keys ['P2', 'P3', 'P4', 'P5']
        """
        # Get features in order from deepest to shallowest
        features = [features_dict[f"C{i}"] for i in range(5, 1, -1)]  # C5, C4, C3, C2

        # Apply lateral connections
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway with lateral connections
        output_features = []
        for i in range(len(laterals)):
            if i == 0:
                # Highest level (C5 -> P5)
                out = laterals[i]
            else:
                # Upsample previous level and add to current lateral
                prev_shape = laterals[i].shape[-2:]
                upsampled = F.interpolate(
                    output_features[-1], size=prev_shape, mode="nearest"
                )
                out = laterals[i] + upsampled

            # Apply output convolution
            out = self.output_convs[i](out)

            # Apply attention module
            out = self.attention_modules[i](out)

            output_features.append(out)

        # Reverse to get P2, P3, P4, P5 order
        output_features = output_features[::-1]

        # Create output dictionary
        fpn_output = {f"P{i}": feat for i, feat in enumerate(output_features, start=2)}

        return fpn_output


def build_custom_backbone_with_fpn(pretrained=True):
    """
    Build complete backbone with FPN for Mask R-CNN.
    Returns backbone and out_channels for RPN/RoI heads.
    """
    # Custom EfficientNet backbone with attention
    backbone = CustomEfficientNetBackbone(pretrained=pretrained)

    # Custom FPN with attention
    in_channels_list = [
        backbone.out_channels["C5"],  # 320
        backbone.out_channels["C4"],  # 112
        backbone.out_channels["C3"],  # 80
        backbone.out_channels["C2"],  # 40
    ]

    fpn = AttentionFPN(in_channels_list, out_channels=256)

    return backbone, fpn, 256  # 256 is FPN output channels


class BackboneWithFPN(nn.Module):
    """Wrapper combining backbone and FPN"""

    def __init__(self, backbone, fpn):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.out_channels = fpn.out_channels

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return fpn_features
