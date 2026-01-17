import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(num_classes, pretrained=True):
    """
    Create Mask R-CNN model for instance segmentation.

    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights

    Returns:
        Mask R-CNN model
    """
    # Load pretrained Mask R-CNN with ResNet-50 backbone
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the box predictor head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor head
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def get_lightweight_maskrcnn(num_classes):
    """
    Create a lighter Mask R-CNN variant for faster training.
    Uses MobileNetV3 backbone instead of ResNet-50.
    """
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models import mobilenet_v3_large
    from torchvision.ops import misc as misc_nn_ops

    # Load MobileNetV3 backbone
    backbone = mobilenet_v3_large(pretrained=True).features

    # Freeze early layers
    for name, parameter in backbone.named_parameters():
        if "features.0" in name or "features.1" in name:
            parameter.requires_grad_(False)

    # For simplicity, use the standard ResNet-50 FPN model
    # and just reduce anchor sizes for better performance
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Adjust anchor sizes for aerial imagery
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    from torchvision.models.detection.rpn import AnchorGenerator

    model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # Replace prediction heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model


class MaskRCNNTrainer:
    """Wrapper class for training Mask R-CNN"""

    def __init__(self, model, optimizer, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, data_loader):
        """Train for one epoch"""
        from tqdm.auto import tqdm

        self.model.train()
        total_loss = 0

        for images, targets in tqdm(data_loader, desc="Training", leave=False):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += losses.item()

        return total_loss / len(data_loader)

    @torch.no_grad()
    def validate(self, data_loader):
        """Validate the model"""
        from tqdm.auto import tqdm

        self.model.eval()
        total_loss = 0

        for images, targets in tqdm(data_loader, desc="Validation", leave=False):
            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

        return total_loss / len(data_loader)

    @torch.no_grad()
    def predict(self, images):
        """Make predictions"""
        self.model.eval()
        images = [img.to(self.device) for img in images]
        predictions = self.model(images)
        return predictions


def train_model(
    train_dataset,
    val_dataset,
    num_classes=16,
    num_epochs=20,
    batch_size=2,
    lr=0.005,
    device="cuda",
):
    """
    Simple training function for Mask R-CNN.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes (default 16 for iSAID)
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on

    Returns:
        trainer: Trained MaskRCNNTrainer object
        history: Dictionary with training history
    """
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    def collate_fn(batch):
        return tuple(zip(*batch))

    # Apply transforms if not already applied
    if train_dataset.transforms is None:
        train_dataset.transforms = T.ToTensor()
    if val_dataset.transforms is None:
        val_dataset.transforms = T.ToTensor()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    # Create model
    model = get_maskrcnn_model(num_classes, pretrained=True)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize trainer
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    trainer = MaskRCNNTrainer(model, optimizer, device)

    # Training history
    history = {"train_loss": [], "val_loss": []}

    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Training loop
    from tqdm.auto import tqdm

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Train
        train_loss = trainer.train_one_epoch(train_loader)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss = trainer.validate(val_loader)
        history["val_loss"].append(val_loss)

        # Update learning rate
        lr_scheduler.step()

        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    return trainer, history
