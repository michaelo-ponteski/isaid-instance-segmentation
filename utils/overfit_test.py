import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import copy

def overfit_single_image_test(model, dataset, idx=0, num_epochs=100, device='cuda'):
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
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = transform(image)

    print(f"\nImage shape: {image.shape}")
    print(f"Number of instances: {len(target['boxes'])}")
    print(f"Classes: {target['labels'].tolist()}")

    # Move to device
    model = model.to(device)
    image = image.to(device)
    target = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in target.items()}

    # Create optimizer - higher LR for aggressive overfitting
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    # Training loop
    model.train()
    losses_history = []

    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Forward pass
        loss_dict = model([image], [target])
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Overfitting Test - Loss Curve')
    plt.grid(True)
    plt.show()

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model([image])

    # Visualize results
    visualize_predictions(image, target, predictions[0], dataset)

    # Check if model learned
    final_loss = losses_history[-1]
    initial_loss = losses_history[0]
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print("\n" + "=" * 80)
    print("RESULTS:")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.2f}%")

    if final_loss < 0.3:
        print("✓ SUCCESS: Model successfully overfitted to the image!")
    elif improvement > 80:
        print("⚠ PARTIAL: Model is learning but may need more epochs")
    else:
        print("✗ FAILURE: Model is not learning properly - check configuration")
    print("=" * 80)

    return losses_history, predictions


def visualize_predictions(image, target, prediction, dataset, conf_threshold=0.5):
    """
    Visualize ground truth and predictions side by side.
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.array(image)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Ground truth
    ax = axes[0]
    ax.imshow(image_np)

    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                fill=False, edgecolor='lime', linewidth=2)
            ax.add_patch(rect)

        # Get category name
        cat_names = dataset.get_category_names()
        cat_name = cat_names.get(int(label), f'Class {label}')

        ax.text(x1, y1-5, f'{cat_name} {score:.2f}',
                   color='white', fontsize=10,
                   bbox=dict(facecolor='lime', alpha=0.7))

    ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Predictions
    ax = axes[1]
    ax.imshow(image_np)

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # Filter by confidence threshold
    keep = scores > conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        cat_names = dataset.get_category_names()
        cat_name = cat_names.get(int(label), f'Class {label}')
        ax.text(x1, y1-5, cat_name,
               color='white', fontsize=10,
               bbox=dict(facecolor='red', alpha=0.7))

    ax.set_title(f'Predictions (conf > {conf_threshold})', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\nPredictions: {len(boxes)} objects detected")


# Complete example workflow
if __name__ == '__main__':
    print("Setting up overfit test...")

    # Import previous components
    from isaid_dataset import iSAIDDataset
    from maskrcnn_model import get_maskrcnn_model

    # Setup
    root_dir = 'iSAID_patches'
    num_classes = 16  # 15 classes + background
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    print("Loading dataset...")
    train_dataset = iSAIDDataset(root_dir, split='train')

    # Create model
    print("Creating model...")
    model = get_maskrcnn_model(num_classes, pretrained=True)

    # Run overfit test
    print("\nStarting overfit test...")
    losses, predictions = overfit_single_image_test(
        model,
        train_dataset,
        idx=0,  # Use first image
        num_epochs=50,
        device=device
    )

    print("\nOverfit test complete!")
    print("If the model successfully overfitted, you're ready to train on the full dataset.")
