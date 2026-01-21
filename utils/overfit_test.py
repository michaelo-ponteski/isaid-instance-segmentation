import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import copy


def overfit_single_image_test(
    model, dataset, idx=0, num_epochs=100, device="cuda", num_images=3
):
    """
    Test if model can overfit to a small set of images.
    This is a sanity check to ensure the model can learn.

    Args:
        model: Mask R-CNN model
        dataset: Dataset object
        idx: Starting index of images to overfit (will use idx, idx+1, idx+2, ...)
        num_epochs: Number of epochs to train
        device: Device to use
        num_images: Number of images to overfit (default: 3)
    """
    print("=" * 80)
    print(f"OVERFIT TEST ({num_images} IMAGES)")
    print("=" * 80)

    # Get multiple images and targets
    images = []
    targets = []

    for i in range(num_images):
        sample_idx = (idx + i) % len(dataset)  # Wrap around if needed
        image, target = dataset[sample_idx]

        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            image = transform(image)

        images.append(image)
        targets.append(target)

        print(f"\nImage {i+1} (idx={sample_idx}):")
        print(f"  Shape: {image.shape}")
        print(f"  Instances: {len(target['boxes'])}")
        print(f"  Classes: {target['labels'].tolist()}")

    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    images = [img.to(device) for img in images]
    targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]

    # Use AdamW for more stable training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.01)

    # Training loop
    model.train()
    losses_history = []

    print(f"\nTraining for {num_epochs} epochs on {num_images} images...")
    for epoch in range(num_epochs):
        # Forward pass with all images as a batch
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Check for NaN or inf
        if torch.isnan(losses) or torch.isinf(losses):
            print(f"Warning: NaN or Inf loss at epoch {epoch+1}. Stopping training.")
            break

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping to prevent explosion (tighter=more stable)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        losses_history.append(losses.item())

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}")
            for k, v in loss_dict.items():
                print(f"  {k}: {v.item():.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overfitting Test - Loss Curve")
    plt.grid(True)
    plt.show()

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Visualize results for all images
    for i, (img, tgt, pred) in enumerate(zip(images, targets, predictions)):
        print(f"\n--- Image {i+1} Results ---")
        num_gt = len(tgt["boxes"])
        visualize_predictions(img, tgt, pred, dataset, num_gt=num_gt)

    # Check if model learned
    final_loss = losses_history[-1]
    initial_loss = losses_history[0]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print("=" * 50)

    return losses_history, predictions


def visualize_predictions(
    image, target, prediction, dataset, conf_threshold=0.5, num_gt=None, mask_alpha=0.4
):
    """
    Visualization of ground truth vs predictions with masks.
    
    Args:
        image: Input image tensor
        target: Ground truth dict with boxes, labels, masks
        prediction: Model prediction dict with boxes, labels, scores, masks
        dataset: Dataset for category names
        conf_threshold: Confidence threshold for predictions
        num_gt: Expected number of ground truth boxes (for printing)
        mask_alpha: Transparency for mask overlay (0-1)
    """
    # RGB colors for different classes (as floats 0-1)
    COLORS = [
        [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0.5, 0], [0.5, 0, 0.5], [0, 1, 1],
        [1, 0, 1], [1, 1, 0], [0.5, 1, 0], [1, 0.5, 0.5], [0.6, 0.3, 0],
        [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0], [1, 0.4, 0.7], [1, 0.84, 0],
    ]
    # String colors for box edges
    COLOR_NAMES = [
        "red", "blue", "green", "orange", "purple", "cyan",
        "magenta", "yellow", "lime", "pink", "brown", "navy",
        "teal", "olive", "coral", "gold",
    ]

    # Convert image to numpy and denormalize if needed
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().permute(1, 2, 0).numpy()
        # Check if image is normalized (values outside 0-1 range)
        if image_np.min() < 0 or image_np.max() > 1:
            # Denormalize using ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
        # Clip to valid range
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = np.array(image) / 255.0 if np.array(image).max() > 1 else np.array(image)

    # Handle Subset wrapper - access underlying dataset
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset
    cat_names = (
        base_dataset.get_category_names()
        if hasattr(base_dataset, "get_category_names")
        else {}
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ground truth with masks
    ax = axes[0]
    img_gt_overlay = image_np.copy()

    # Overlay ground truth masks
    if "masks" in target and len(target["masks"]) > 0:
        gt_masks = target["masks"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        for mask, label in zip(gt_masks, gt_labels):
            color = np.array(COLORS[int(label) % len(COLORS)])
            # Handle different mask formats (H,W) or (1,H,W)
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > 0.5
            img_gt_overlay[mask_bool] = (
                img_gt_overlay[mask_bool] * (1 - mask_alpha) + color * mask_alpha
            )

    ax.imshow(img_gt_overlay)

    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            color = COLOR_NAMES[int(label) % len(COLOR_NAMES)]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
            )
            ax.add_patch(rect)

    num_gt_masks = len(target["masks"]) if "masks" in target else 0
    ax.set_title(f"Ground Truth ({len(target['boxes'])} boxes, {num_gt_masks} masks)")
    ax.axis("off")

    # Predictions with masks
    ax = axes[1]
    img_pred_overlay = image_np.copy()

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    pred_masks = prediction["masks"].cpu().numpy() if "masks" in prediction else None

    # Filter by confidence
    keep = scores > conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    if pred_masks is not None:
        pred_masks = pred_masks[keep]

    # Overlay predicted masks
    if pred_masks is not None and len(pred_masks) > 0:
        for mask, label in zip(pred_masks, labels):
            color = np.array(COLORS[int(label) % len(COLORS)])
            # Predicted masks are (1, H, W) with probabilities
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > 0.5
            img_pred_overlay[mask_bool] = (
                img_pred_overlay[mask_bool] * (1 - mask_alpha) + color * mask_alpha
            )

    ax.imshow(img_pred_overlay)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = COLOR_NAMES[int(label) % len(COLOR_NAMES)]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f"{score:.2f}", color=color, fontsize=8)

    num_pred_masks = len(pred_masks) if pred_masks is not None else 0
    ax.set_title(f"Predictions ({len(boxes)} boxes, {num_pred_masks} masks, conf>{conf_threshold})")
    ax.axis("off")

    # Legend for classes that appear
    used_labels = set(target["labels"].cpu().numpy().tolist())
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=COLOR_NAMES[int(l) % len(COLOR_NAMES)],
            linewidth=2,
            label=cat_names.get(int(l), f"Class {l}"),
        )
        for l in used_labels
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=min(len(used_labels), 5)
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Print comparison
    if num_gt is not None:
        print(f"\nFound {len(boxes)} boxes (should be {num_gt})")
        print(f"Found {num_pred_masks} masks (should be {num_gt_masks})")
