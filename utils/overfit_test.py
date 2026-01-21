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
    image, target, prediction, dataset, conf_threshold=0.5, num_gt=None
):
    """
    Simple visualization of ground truth vs predictions.
    """
    # Colors for different classes
    COLORS = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "lime",
        "pink",
        "brown",
        "navy",
        "teal",
        "olive",
        "coral",
        "gold",
    ]

    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)

    # Handle Subset wrapper - access underlying dataset
    if hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset
    cat_names = base_dataset.get_category_names() if hasattr(base_dataset, "get_category_names") else {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    ax = axes[0]
    ax.imshow(image_np)

    if "boxes" in target and len(target["boxes"]) > 0:
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            color = COLORS[int(label) % len(COLORS)]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
            )
            ax.add_patch(rect)

    ax.set_title(f"Ground Truth ({len(target['boxes'])} boxes)")
    ax.axis("off")

    # Predictions
    ax = axes[1]
    ax.imshow(image_np)

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    # Filter by confidence
    keep = scores > conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color = COLORS[int(label) % len(COLORS)]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f"{score:.2f}", color=color, fontsize=8)

    ax.set_title(f"Predictions ({len(boxes)} boxes, conf>{conf_threshold})")
    ax.axis("off")

    # Legend for classes that appear
    used_labels = set(target["labels"].cpu().numpy().tolist())
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=COLORS[int(l) % len(COLORS)],
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
