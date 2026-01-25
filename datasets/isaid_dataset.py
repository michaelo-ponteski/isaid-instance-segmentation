import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


class iSAIDDataset(Dataset):
    """
    Dataset loader for iSAID instance segmentation dataset.
    Supports train/val/test splits with COCO-format annotations.

    Training augmentations (flip, color jitter) are applied automatically
    for the 'train' split when augment=True. Augmentations are annotation-safe,
    meaning bounding boxes and masks are transformed together with the image.

    Dataset Filtering (applied during initialization, before training):
    - max_boxes_per_image: Images with more boxes are excluded (prevents RAM/VRAM spikes)
    - max_empty_fraction: Controls ratio of images with 0 boxes (prevents empty-dominated batches)

    All filtering is deterministic and happens BEFORE training starts.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        transforms=None,
        filter_empty=False,  # Changed default: now we control empty images via max_empty_fraction
        image_size=800,
        augment=None,
        # === NEW: Outlier filtering parameters ===
        max_boxes_per_image=400,  # Images with more boxes are excluded
        max_empty_fraction=0.3,   # Max fraction of images with 0 boxes (0.3 = 30%)
        filter_seed=42,           # Seed for deterministic empty image sampling
        verbose=True,             # Print dataset statistics report
    ):
        """
        Args:
            root_dir: Path to iSAID_patches directory
            split: 'train', 'val', or 'test'
            transforms: Optional transforms to apply (normalization/tensor conversion)
            filter_empty: If True, remove ALL images without annotations (legacy behavior)
            image_size: Target size for images (images will be resized to square)
            augment: If True, apply training augmentations. If None, defaults to
                     True for train split, False otherwise.
            max_boxes_per_image: Maximum allowed boxes per image. Images exceeding
                     this limit are excluded to prevent memory spikes and slow batches.
                     Set to None to disable filtering. Default: 400
            max_empty_fraction: Maximum fraction of empty images (0 boxes) in dataset.
                     Excess empty images are randomly sampled out (deterministically).
                     Set to 1.0 to keep all empty images. Default: 0.3 (30%)
            filter_seed: Random seed for deterministic empty image sampling.
            verbose: If True, print detailed dataset statistics after initialization.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.filter_empty = filter_empty
        self.image_size = image_size
        self.max_boxes_per_image = max_boxes_per_image
        self.max_empty_fraction = max_empty_fraction
        self.filter_seed = filter_seed
        self.verbose = verbose

        # Set augmentation flag: default to True for train, False for val/test
        if augment is None:
            self.augment = split == "train"
        else:
            self.augment = augment

        # Initialize augmentation pipeline for training
        self.train_augmentation = None
        if self.augment:
            from training.transforms import get_train_augmentation

            self.train_augmentation = get_train_augmentation()

        # Setup paths
        self.img_dir = os.path.join(root_dir, split, "images")

        # Load annotations if not test
        self.annotations = None
        self.images_info = []
        self.anns_per_image = {}

        # Statistics tracking (for reporting)
        self._stats = {
            "total_images_original": 0,
            "total_boxes_original": 0,
            "rejected_too_many_boxes": 0,
            "rejected_empty_excess": 0,
            "empty_images_kept": 0,
            "non_empty_images_kept": 0,
        }

        if split != "test":
            ann_file = os.path.join(
                root_dir, split, f"instances_only_filtered_{split}.json"
            )
            with open(ann_file, "r") as f:
                self.annotations = json.load(f)

            # Create image id to annotations mapping
            all_images_info = self.annotations["images"]
            self._stats["total_images_original"] = len(all_images_info)

            for ann in self.annotations["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self.anns_per_image:
                    self.anns_per_image[img_id] = []
                self.anns_per_image[img_id].append(ann)

            self._stats["total_boxes_original"] = len(self.annotations["annotations"])

            # === FILTERING PIPELINE ===
            # All filtering is done BEFORE training, deterministically

            # Step 1: Legacy filter_empty (removes ALL empty images if True)
            if self.filter_empty:
                all_images_info = [
                    img
                    for img in all_images_info
                    if img["id"] in self.anns_per_image
                    and len(self.anns_per_image[img["id"]]) > 0
                ]

            # Step 2: Filter images with too many boxes (outliers)
            # Why: Images with 400+ boxes can cause:
            #   - Memory spikes (each box = proposals, ROI features)
            #   - Slow batches (batch time dominated by densest image)
            #   - Training instability (noisy gradients from crowded scenes)
            if self.max_boxes_per_image is not None:
                filtered_images = []
                for img in all_images_info:
                    num_boxes = len(self.anns_per_image.get(img["id"], []))
                    if num_boxes <= self.max_boxes_per_image:
                        filtered_images.append(img)
                    else:
                        self._stats["rejected_too_many_boxes"] += 1
                all_images_info = filtered_images

            # Step 3: Control empty image fraction
            # Why: Too many empty images can cause:
            #   - Batches dominated by empty samples (poor gradient signal)
            #   - Wasted computation on background-only images
            # But we KEEP some empty images because:
            #   - They teach the model to recognize background
            #   - They reduce false positives
            if self.max_empty_fraction < 1.0:
                empty_images = []
                non_empty_images = []

                for img in all_images_info:
                    num_boxes = len(self.anns_per_image.get(img["id"], []))
                    if num_boxes == 0:
                        empty_images.append(img)
                    else:
                        non_empty_images.append(img)

                # Calculate how many empty images to keep
                total_non_empty = len(non_empty_images)
                # max_empty_fraction = empty / (empty + non_empty)
                # => empty = max_empty_fraction * (empty + non_empty)
                # => empty = max_empty_fraction * total / (1 - max_empty_fraction) when non_empty is fixed
                if self.max_empty_fraction > 0 and total_non_empty > 0:
                    max_empty = int(
                        (self.max_empty_fraction / (1 - self.max_empty_fraction))
                        * total_non_empty
                    )
                else:
                    max_empty = 0

                # Deterministically sample empty images if we have too many
                if len(empty_images) > max_empty:
                    rng = random.Random(self.filter_seed)
                    rng.shuffle(empty_images)
                    rejected_count = len(empty_images) - max_empty
                    empty_images = empty_images[:max_empty]
                    self._stats["rejected_empty_excess"] = rejected_count

                self._stats["empty_images_kept"] = len(empty_images)
                self._stats["non_empty_images_kept"] = len(non_empty_images)

                # Combine: non-empty first, then empty (order doesn't matter, shuffle in DataLoader)
                all_images_info = non_empty_images + empty_images
            else:
                # Count stats even if not filtering
                for img in all_images_info:
                    num_boxes = len(self.anns_per_image.get(img["id"], []))
                    if num_boxes == 0:
                        self._stats["empty_images_kept"] += 1
                    else:
                        self._stats["non_empty_images_kept"] += 1

            self.images_info = all_images_info

            # Print statistics report
            if self.verbose:
                self._print_statistics_report()

        else:
            # For test set, just list all images (no filtering)
            img_files = [
                f
                for f in os.listdir(self.img_dir)
                if f.endswith(".png") and "instance" not in f
            ]
            self.images_info = [
                {"file_name": f, "id": i} for i, f in enumerate(img_files)
            ]
            if self.verbose:
                print(f"[{self.split.upper()}] Test set: {len(self.images_info)} images")

    def _print_statistics_report(self):
        """Print detailed dataset statistics report after filtering."""
        # Compute box count statistics for final dataset
        box_counts = [
            len(self.anns_per_image.get(img["id"], []))
            for img in self.images_info
        ]

        if len(box_counts) == 0:
            print(f"[{self.split.upper()}] WARNING: Dataset is empty after filtering!")
            return

        box_counts = np.array(box_counts)

        print("\n" + "=" * 60)
        print(f"DATASET STATISTICS: {self.split.upper()}")
        print("=" * 60)

        # Original vs final counts
        print(f"\nImage Counts:")
        print(f"   Original images:        {self._stats['total_images_original']}")
        print(f"   Final images:           {len(self.images_info)}")

        # Rejection reasons
        total_rejected = (
            self._stats["rejected_too_many_boxes"] +
            self._stats["rejected_empty_excess"]
        )
        if total_rejected > 0:
            print(f"\nRejected Images ({total_rejected} total):")
            if self._stats["rejected_too_many_boxes"] > 0:
                print(f"   - Too many boxes (>{self.max_boxes_per_image}): "
                      f"{self._stats['rejected_too_many_boxes']}")
            if self._stats["rejected_empty_excess"] > 0:
                print(f"   - Empty image excess:       "
                      f"{self._stats['rejected_empty_excess']}")

        # Empty image statistics
        total_final = len(self.images_info)
        empty_pct = (self._stats["empty_images_kept"] / total_final * 100) if total_final > 0 else 0
        print(f"\nBox Distribution (final dataset):")
        print(f"   Empty images (0 boxes):  {self._stats['empty_images_kept']} "
              f"({empty_pct:.1f}%)")
        print(f"   Non-empty images:        {self._stats['non_empty_images_kept']}")

        # Box count statistics
        print(f"\nBox Count Statistics:")
        print(f"   Min:    {box_counts.min()}")
        print(f"   Max:    {box_counts.max()}")
        print(f"   Mean:   {box_counts.mean():.1f}")
        print(f"   Median: {np.median(box_counts):.1f}")
        print(f"   Std:    {box_counts.std():.1f}")

        # Percentiles
        print(f"\n   Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"     {p}th: {np.percentile(box_counts, p):.0f}")

        print("=" * 60 + "\n")

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images_info[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Prepare target dict
        target = {"image_id": img_info["id"]}

        if self.split != "test":
            img_id = img_info["id"]
            anns = self.anns_per_image.get(img_id, [])

            # Extract boxes, labels, masks
            boxes = []
            labels = []
            masks = []
            areas = []

            for ann in anns:
                # Bounding box [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
                areas.append(ann["area"])

                # Segmentation mask - convert polygon to binary mask
                if "segmentation" in ann and len(ann["segmentation"]) > 0:
                    mask = self._polygon_to_mask(
                        ann["segmentation"], img_info["height"], img_info["width"]
                    )
                    masks.append(mask)
                else:
                    # Empty mask fallback
                    mask = np.zeros(
                        (img_info["height"], img_info["width"]), dtype=np.uint8
                    )
                    masks.append(mask)

            if len(boxes) > 0:
                boxes = np.array(boxes, dtype=np.float32)
                masks = np.array(masks, dtype=np.uint8)
                labels = np.array(labels, dtype=np.int64)
                areas = np.array(areas, dtype=np.float32)

                # Resize image and adjust boxes/masks if image_size is specified
                if self.image_size is not None:
                    # Calculate scale factors
                    scale_x = self.image_size / orig_width
                    scale_y = self.image_size / orig_height

                    # Resize image
                    image = image.resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )

                    # Scale boxes
                    boxes[:, [0, 2]] *= scale_x  # x coordinates
                    boxes[:, [1, 3]] *= scale_y  # y coordinates

                    # Resize masks
                    resized_masks = []
                    for mask in masks:
                        mask_img = Image.fromarray(mask)
                        resized_mask = mask_img.resize(
                            (self.image_size, self.image_size), Image.NEAREST
                        )
                        resized_masks.append(np.array(resized_mask))
                    masks = np.array(resized_masks, dtype=np.uint8)

                # Apply training augmentations (flip, color jitter)
                # This must be done AFTER resizing but BEFORE validation filtering
                # to ensure boxes and masks are transformed together with the image.
                # Augmentations are only applied for training split (controlled by
                # self.train_augmentation being set).
                if self.train_augmentation is not None:
                    image, boxes, masks = self.train_augmentation(image, boxes, masks)

                # Get current image dimensions for validation
                if isinstance(image, Image.Image):
                    img_w, img_h = image.size
                else:
                    img_h, img_w = image.shape[:2]

                # Filter out invalid boxes (zero or negative width/height)
                # This prevents NaN loss during training
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                min_size = 1.0  # Minimum box size in pixels
                valid_mask = (widths >= min_size) & (heights >= min_size)

                # Also filter out boxes outside image bounds
                valid_mask &= (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
                valid_mask &= (boxes[:, 2] <= img_w) & (boxes[:, 3] <= img_h)

                # Clamp boxes to image bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

                if valid_mask.sum() > 0:
                    boxes = boxes[valid_mask]
                    labels = labels[valid_mask]
                    masks = masks[valid_mask]
                    areas = areas[valid_mask]

                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
                target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
                target["area"] = torch.as_tensor(areas, dtype=torch.float32)
                target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
            else:
                # No annotations
                target_height = (
                    self.image_size
                    if self.image_size is not None
                    else img_info["height"]
                )
                target_width = (
                    self.image_size
                    if self.image_size is not None
                    else img_info["width"]
                )
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
                target["masks"] = torch.zeros(
                    (0, target_height, target_width), dtype=torch.uint8
                )

        # Resize image for test split too
        if self.split == "test" and self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.transforms:
            image = self.transforms(image)
        else:
            # Convert PIL Image to tensor if no transforms provided
            image = transforms.ToTensor()(image)

        return image, target

    def _polygon_to_mask(self, segmentation, height, width):
        """Convert polygon segmentation to binary mask"""
        import cv2

        mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in segmentation:
            # Polygon is [x1, y1, x2, y2, ...] - reshape to [[x1, y1], [x2, y2], ...]
            poly_array = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [poly_array], 1)

        return mask

    def get_category_names(self):
        """Returns mapping of category IDs to names"""
        if self.annotations:
            return {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        return {}


# Example usage and visualization
def visualize_sample(dataset, idx):
    """Visualize a sample from the dataset"""
    image, target = dataset[idx]

    plt.figure(figsize=(12, 8))

    # Convert tensor to numpy for visualization and denormalize if needed
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        # Check if image is normalized (values outside 0-1 range)
        if image_np.min() < 0 or image_np.max() > 1:
            # Denormalize using ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
        # Clip to valid range
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = np.array(image)

    plt.imshow(image_np)

    # Draw bounding boxes
    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        ax = plt.gca()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
            )
            ax.add_patch(rect)
            # Get category name if available - handle Subset wrapper
            if hasattr(dataset, "dataset"):
                base_dataset = dataset.dataset
            else:
                base_dataset = dataset
            cat_names = (
                base_dataset.get_category_names()
                if hasattr(base_dataset, "get_category_names")
                else {}
            )
            cat_name = cat_names.get(int(label), f"Class {label}")

            plt.text(
                x1,
                y1 - 5,
                cat_name,
                color="white",
                fontsize=10,
                bbox=dict(facecolor="red", alpha=0.7),
            )

    plt.axis("off")
    plt.title(f"Sample {idx}")
    plt.tight_layout()
    plt.show()


# Initialize datasets
if __name__ == "__main__":
    root_dir = "iSAID_patches"

    # Create datasets with filtering enabled
    # - max_boxes_per_image=400: Exclude extreme outliers
    # - max_empty_fraction=0.3: Keep max 30% empty images
    print("Creating datasets with outlier filtering...\n")
    
    train_dataset = iSAIDDataset(
        root_dir, 
        split="train",
        max_boxes_per_image=400,
        max_empty_fraction=0.3,
        filter_seed=42,
    )
    
    val_dataset = iSAIDDataset(
        root_dir, 
        split="val",
        max_boxes_per_image=400,
        max_empty_fraction=0.3,
        filter_seed=42,
    )
    
    test_dataset = iSAIDDataset(root_dir, split="test")

    print(f"Final dataset sizes:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Get category names
    categories = train_dataset.get_category_names()
    print(f"\nCategories: {categories}")

    # Visualize a sample
    visualize_sample(train_dataset, 0)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Test iteration
    images, targets = next(iter(train_loader))
    print(f"\nBatch size: {len(images)}")
    print(
        f"Image shape: {images[0].shape if isinstance(images[0], torch.Tensor) else 'PIL Image'}"
    )
    print(f"Target keys: {targets[0].keys()}")
