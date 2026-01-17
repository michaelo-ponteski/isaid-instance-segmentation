import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def overfit_single_image_test(model, dataset, idx=0, num_epochs=100, device="cuda"):
    """
    Test if model can overfit to a single image - a sanity check to ensure the model can learn.

    Args:
        model: Mask R-CNN model
        dataset: Dataset object
        idx: Index of image to overfit
        num_epochs: Number of epochs to train
        device: Device to use

    Returns:
        losses_history: List of loss values
        prediction: Final prediction on the image
    """
    print("=" * 60)
    print("OVERFIT TEST - Single Image")
    print("=" * 60)

    # Get single image and target
    image, target = dataset[idx]

    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)

    print(f"Image shape: {image.shape}")
    print(f"Number of instances: {len(target['boxes'])}")

    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)
    target = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()
    }

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    model.train()
    losses_history = []

    for epoch in range(num_epochs):
        # Forward pass
        loss_dict = model([image], [target])
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        losses_history.append(losses.item())

        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overfit Test - Loss Curve")
    plt.grid(True)
    plt.show()

    # Make predictions
    model.eval()
    with torch.no_grad():
        prediction = model([image])[0]

    # Show results
    final_loss = losses_history[-1]
    initial_loss = losses_history[0]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nInitial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print("=" * 60)

    # Visualize
    visualize_overfit_result(image, target, prediction, dataset)

    return losses_history, prediction


def visualize_overfit_result(image, target, prediction, dataset, conf_threshold=0.5):
    """Visualize ground truth vs predictions for overfit test."""
    COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
        ax.text(x1, y1 - 3, f"{score:.2f}", color=color, fontsize=10)

    ax.set_title(f"Predictions ({len(boxes)} boxes, conf>{conf_threshold})")
    ax.axis("off")

    plt.tight_layout()
    plt.show()
