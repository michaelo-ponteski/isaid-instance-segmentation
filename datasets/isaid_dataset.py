import os
import json
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
    """

    def __init__(self, root_dir, split="train", transforms=None):
        """
        Args:
            root_dir: Path to iSAID_patches directory
            split: 'train', 'val', or 'test'
            transforms: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        # Setup paths
        self.img_dir = os.path.join(root_dir, split, "images")

        # Load annotations if not test
        self.annotations = None
        self.images_info = []
        self.anns_per_image = {}

        if split != "test":
            ann_file = os.path.join(
                root_dir, split, f"instance_only_filtered_{split}.json"
            )
            with open(ann_file, "r") as f:
                self.annotations = json.load(f)

            # Create image id to annotations mapping
            self.images_info = self.annotations["images"]
            for ann in self.annotations["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self.anns_per_image:
                    self.anns_per_image[img_id] = []
                self.anns_per_image[img_id].append(ann)
        else:
            # For test set, just list all images
            img_files = [
                f
                for f in os.listdir(self.img_dir)
                if f.endswith(".png") and "instance" not in f
            ]
            self.images_info = [
                {"file_name": f, "id": i} for i, f in enumerate(img_files)
            ]

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images_info[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # Load image
        image = Image.open(img_path).convert("RGB")

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
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
                target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                target["area"] = torch.as_tensor(areas, dtype=torch.float32)
                target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
            else:
                # No annotations
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros(0, dtype=torch.int64)
                target["masks"] = torch.zeros(
                    (0, img_info["height"], img_info["width"]), dtype=torch.uint8
                )

        if self.transforms:
            image = self.transforms(image)

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

    # Convert tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
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
            # Get category name if available
            cat_names = dataset.get_category_names()
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
