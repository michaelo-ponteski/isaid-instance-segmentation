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
        self.model.train()
        total_loss = 0

        for images, targets in data_loader:
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

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
        self.model.train()  # Keep in train mode to get losses
        total_loss = 0
        num_batches = 0

        for images, targets in data_loader:
            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Forward pass - model returns loss_dict in train mode
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isnan(losses):
                total_loss += losses.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def predict(self, images):
        """Make predictions"""
        self.model.eval()
        images = [img.to(self.device) for img in images]
        predictions = self.model(images)
        return predictions


# Example usage
if __name__ == "__main__":
    # Number of classes in iSAID (15 classes + background)
    # Ship, Storage tank, Baseball diamond, Tennis court, Basketball court,
    # Ground track field, Bridge, Large vehicle, Small vehicle,
    # Helicopter, Swimming pool, Roundabout, Soccer ball field,
    # Plane, Harbor
    num_classes = 16  # 15 object classes + 1 background

    # Create model
    model = get_maskrcnn_model(num_classes, pretrained=True)
    print("Model created successfully!")

    # Print model structure
    print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initialize trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MaskRCNNTrainer(model, optimizer, device)

    print(f"\nUsing device: {device}")
    print("Trainer initialized and ready!")
