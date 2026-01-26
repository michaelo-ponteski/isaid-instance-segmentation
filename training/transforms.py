"""
Transforms for training and validation.

Augmentation Strategy:
- Training: RandomHorizontalFlip + ColorJitter + Normalization
- Validation: Normalization only (no augmentation)

NOTE: Spatial augmentations (flip) must be applied together with bounding boxes
and masks to maintain annotation consistency. This is handled in the dataset's
__getitem__ method via the AugmentationTransform class.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
import random


class Transform:
    """Wrapper for albumentations transforms (normalization only)."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        transformed = self.transform(image=image)
        return transformed["image"]


class TrainAugmentation:
    """
    Training augmentations that handle image, boxes, and masks together.

    Augmentations applied:
    1. RandomHorizontalFlip (p=0.5): Flips image and correctly transforms
       bounding boxes and instance masks.
    2. ColorJitter: Mild brightness/contrast perturbations (image-only,
       no effect on annotations).

    This class must be called BEFORE the normalization transform.
    """

    def __init__(self, flip_prob=0.5, brightness=0.1, contrast=0.1):
        """
        Args:
            flip_prob: Probability of horizontal flip (default 0.5)
            brightness: Max brightness jitter factor (default 0.1 = ±10%)
            contrast: Max contrast jitter factor (default 0.1 = ±10%)
        """
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, boxes, masks):
        """
        Apply augmentations to image, boxes, and masks.

        Args:
            image: PIL Image or numpy array (H, W, C)
            boxes: numpy array of shape (N, 4) in [x1, y1, x2, y2] format
            masks: numpy array of shape (N, H, W) - binary instance masks

        Returns:
            Tuple of (augmented_image, augmented_boxes, augmented_masks)
            - image: numpy array (H, W, C)
            - boxes: numpy array (N, 4)
            - masks: numpy array (N, H, W)
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Make copies to avoid modifying originals
        image = image.copy()
        boxes = boxes.copy() if len(boxes) > 0 else boxes
        masks = masks.copy() if len(masks) > 0 else masks

        height, width = image.shape[:2]

        # 1. Random Horizontal Flip
        if random.random() < self.flip_prob:
            # Flip image horizontally
            image = np.fliplr(image).copy()

            if len(boxes) > 0:
                # Flip boxes: x1_new = width - x2_old, x2_new = width - x1_old
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = width - x2  # new x1
                boxes[:, 2] = width - x1  # new x2

            if len(masks) > 0:
                # Flip each mask horizontally (along width axis)
                masks = np.flip(masks, axis=2).copy()

        # 2. Color Jitter (brightness and contrast only)
        # These don't affect annotations, only the image pixels
        image = self._apply_color_jitter(image)

        return image, boxes, masks

    def _apply_color_jitter(self, image):
        """
        Apply mild brightness and contrast jitter.

        Uses conservative ranges to avoid unrealistic color distortions
        while still providing useful augmentation for generalization.
        """
        # Convert to float for manipulation
        image = image.astype(np.float32)

        # Brightness: multiply by factor in [1-brightness, 1+brightness]
        if self.brightness > 0:
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = image * brightness_factor

        # Contrast: adjust around mean
        if self.contrast > 0:
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean

        # Clip to valid range and convert back to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image


def get_transforms(train=True):
    """
    Get transforms for training or validation.

    For training:
        - Augmentations (flip, color jitter) are applied in the dataset
          via TrainAugmentation class to handle boxes/masks correctly.
        - This transform only handles normalization and tensor conversion.

    For validation:
        - No augmentations, only normalization and tensor conversion.

    Returns:
        Transform object for normalization/tensor conversion
    """
    # Both train and val use the same normalization transform here
    # Training augmentations are handled separately in the dataset
    return Transform(
        A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    )


def get_train_augmentation():
    """
    Get the training augmentation pipeline.

    Returns:
        TrainAugmentation object that handles spatial augmentations
        (flip) together with bounding boxes and masks.
    """
    return TrainAugmentation(
        flip_prob=0.5,  # 50% chance of horizontal flip
        brightness=0.2,  # ±10% brightness variation
        contrast=0.2,  # ±10% contrast variation
    )


def get_visualization_transform():
    """Get transform for visualization (tensor only, no normalization)."""
    return Transform(
        A.Compose(
            [
                ToTensorV2(),
            ]
        )
    )
