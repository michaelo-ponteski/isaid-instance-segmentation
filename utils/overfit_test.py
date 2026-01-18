import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import copy


def overfit_single_image_test(model, dataset, idx=0, num_epochs=100, device="cuda"):
    """
    Test if model can overfit to a single image.
    This is a sanity check to ensure the model can learn.

    Args:
        model: Mask R-CNN model
        dataset: Dataset object
        idx: Index of image to overfit
        num_epochs: Number of epochs to train
        device: Device to use
    """
    print("=" * 80)
    print("OVERFIT SINGLE IMAGE TEST")
    print("=" * 80)

    # Get single image and target
    image, target = dataset[idx]

    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        image = transform(image)

    print(f"\nImage shape: {image.shape}")
    print(f"Number of instances: {len(target['boxes'])}")
    print(f"Classes: {target['labels'].tolist()}")

    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)
    target = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()
    }

    # Use AdamW for more stable training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.01)

    # Training loop
    model.train()
    losses_history = []

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Forward pass
        loss_dict = model([image], [target])
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
        predictions = model([image])

    # Visualize results
    num_gt = len(target["boxes"])
    visualize_predictions(image, target, predictions[0], dataset, num_gt=num_gt)

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

    cat_names = dataset.get_category_names()
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
