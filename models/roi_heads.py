"""
Custom RoI Heads for Mask R-CNN with additional layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign


class CustomMaskHead(nn.Module):
    """
    Custom mask prediction head with additional convolutional layers.
    """

    def __init__(self, in_channels, hidden_dim, num_classes):
        super().__init__()

        # Increased number of convolutional layers (own layers)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_dim)

        # Additional residual connection (własna warstwa)
        self.residual = nn.Conv2d(in_channels, hidden_dim, 1)

        # Deconvolution for upsampling
        self.deconv = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)
        self.bn_deconv = nn.BatchNorm2d(hidden_dim)

        # Final prediction layer
        self.mask_fcn_logits = nn.Conv2d(hidden_dim, num_classes, 1)

        # Initialize weights
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Store input for residual
        identity = self.residual(x)

        # Convolutional blocks with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Add residual connection
        x = x + identity
        x = F.relu(x)

        # Upsample with deconvolution
        x = F.relu(self.bn_deconv(self.deconv(x)))

        # Final mask prediction
        x = self.mask_fcn_logits(x)

        return x


class CustomBoxFeatureExtractor(nn.Module):
    """
    Custom box feature extractor with additional FC layers.
    This extracts representation features from pooled RoI features.
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        # Additional FC layers (własne warstwy)
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Initialize
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)

        self.out_channels = representation_size

    def forward(self, x):
        # Flatten
        x = x.flatten(start_dim=1)

        # FC layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x


class CustomBoxPredictor(nn.Module):
    """
    Custom box predictor that outputs class scores and bbox deltas.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Additional layer before prediction
        self.fc = nn.Linear(in_channels, in_channels // 2)
        self.dropout = nn.Dropout(0.3)

        # Classification and bbox regression heads
        self.cls_score = nn.Linear(in_channels // 2, num_classes)
        self.bbox_pred = nn.Linear(in_channels // 2, num_classes * 4)

        # Initialize
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc.bias, 0)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class CustomBoxHead(nn.Module):
    """
    Custom box prediction head with additional FC layers.
    Więcej warstw niż standardowy FastRCNNPredictor
    """

    def __init__(self, in_channels, representation_size, num_classes):
        super().__init__()

        # Additional FC layers (własne warstwy)
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc3 = nn.Linear(representation_size, representation_size // 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Classification and bbox regression heads
        self.cls_score = nn.Linear(representation_size // 2, num_classes)
        self.bbox_pred = nn.Linear(representation_size // 2, num_classes * 4)

        # Initialize
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Flatten
        x = x.flatten(start_dim=1)

        # FC layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))

        # Predictions
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def build_custom_roi_heads(
    box_roi_pool, mask_roi_pool, box_head, mask_head, num_classes
):
    """
    Helper to organize RoI heads.
    Note: Full RoI heads integration requires modifying torchvision's RoIHeads class
    or using the complete custom implementation below.
    """
    return {
        "box_roi_pool": box_roi_pool,
        "mask_roi_pool": mask_roi_pool,
        "box_head": box_head,
        "mask_head": mask_head,
    }


# Configuration for RoI pooling
def get_roi_pooling_config():
    """
    Configuration for Multi-scale RoI Align.
    Used in Mask R-CNN for both box and mask prediction.
    """
    # Box RoI pooling
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["P2", "P3", "P4", "P5"],  # FPN levels
        output_size=7,  # 7x7 for box head
        sampling_ratio=2,
    )

    # Mask RoI pooling
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["P2", "P3", "P4", "P5"],  # FPN levels
        output_size=14,  # 14x14 for mask head
        sampling_ratio=2,
    )

    return box_roi_pool, mask_roi_pool
