"""
Simple transforms for training and validation.

NOTE: Spatial augmentations (flip, rotate) are applied in the dataset's __getitem__ 
to ensure bounding boxes and masks are transformed together with the image.
These transforms only handle normalization and tensor conversion.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


class Transform:
    """Wrapper for albumentations transforms (normalization only)."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        transformed = self.transform(image=image)
        return transformed["image"]


def get_transforms(train=True):
    """
    Get transforms for training or validation.
    
    NOTE: Only normalization is applied here. Spatial augmentations (flip, rotate)
    should be applied in the dataset where boxes/masks can be transformed together.
    """
    # Both train and val use the same normalization-only transform
    # Spatial augmentations are handled separately in the dataset
    return Transform(
        A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    )


def get_visualization_transform():
    """Get transform for visualization (tensor only, no normalization)."""
    return Transform(
        A.Compose([
            ToTensorV2(),
        ])
    )
