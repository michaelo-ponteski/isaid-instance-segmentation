import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def plot_training_history(history_path=None, history_dict=None):
    """
    Plot training history (loss curves)

    Args:
        history_path: Path to JSON file with history
        history_dict: Or directly provide history dictionary
    """
    if history_path:
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = history_dict

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot
    if "learning_rates" in history and len(history["learning_rates"]) > 0:
        ax = axes[1]
        ax.plot(epochs, history["learning_rates"], "g-", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs: {len(history['train_loss'])}")
    print(
        f"Best train loss: {min(history['train_loss']):.4f} (epoch {np.argmin(history['train_loss']) + 1})"
    )
    print(
        f"Best val loss: {min(history['val_loss']):.4f} (epoch {np.argmin(history['val_loss']) + 1})"
    )
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print("=" * 60)


@torch.no_grad()
def visualize_predictions_batch(
    model, dataset, device="cuda", num_samples=4, conf_threshold=0.5
):
    """
    Visualize predictions on multiple samples

    Args:
        model: Trained Mask R-CNN model
        dataset: Dataset to sample from
        device: Device to run inference on
        num_samples: Number of samples to visualize
        conf_threshold: Confidence threshold for predictions
    """
    model.eval()
    model.to(device)

    # Sample random indices
    indices = np.random.choice(
        len(dataset), size=min(num_samples, len(dataset)), replace=False
    )

    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(indices):
        image, target = dataset[sample_idx]

        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            from torchvision import transforms

            image = transforms.ToTensor()(image)

        # Get predictions
        image_gpu = image.to(device)
        predictions = model([image_gpu])[0]

        # Convert image to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Ground truth
        ax = axes[idx, 0]
        ax.imshow(image_np)

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].cpu().numpy()
            labels = target["labels"].cpu().numpy()
            cat_names = dataset.get_category_names()

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
                ax.add_patch(rect)

                cat_name = cat_names.get(int(label), f"Class {label}")
                ax.text(
                    x1,
                    y1 - 5,
                    cat_name,
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="lime", alpha=0.7),
                )

        ax.set_title(
            f"Ground Truth (Sample {sample_idx})", fontsize=12, fontweight="bold"
        )
        ax.axis("off")

        # Predictions
        ax = axes[idx, 1]
        ax.imshow(image_np)

        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        # Filter by confidence
        keep = scores > conf_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        cat_names = dataset.get_category_names()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
            )
            ax.add_patch(rect)

            cat_name = cat_names.get(int(label), f"Class {label}")
            ax.text(
                x1,
                y1 - 5,
                f"{cat_name} {score:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.7),
            )

        ax.set_title(
            f"Predictions ({len(boxes)} detections)", fontsize=12, fontweight="bold"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("MODEL PARAMETERS")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60)

    return total_params, trainable_params


def estimate_training_time(train_loader, val_loader, num_epochs, sample_batches=5):
    """
    Estimate total training time by timing a few batches

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        sample_batches: Number of batches to time for estimation
    """
    import time

    print("Estimating training time...")
    print(f"Sampling {sample_batches} batches...")

    # Time training batches
    times = []
    for i, (images, targets) in enumerate(train_loader):
        if i >= sample_batches:
            break
        start = time.time()
        # Simulate processing (just measure data loading)
        _ = [img.shape for img in images]
        times.append(time.time() - start)

    avg_batch_time = np.mean(times)
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)

    # Estimate (assuming val is ~0.5x speed of train)
    estimated_epoch_time = avg_batch_time * (
        total_train_batches + 0.5 * total_val_batches
    )
    estimated_total_time = estimated_epoch_time * num_epochs

    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATE")
    print("=" * 60)
    print(f"Average batch time: {avg_batch_time:.3f}s")
    print(f"Training batches per epoch: {total_train_batches}")
    print(f"Validation batches per epoch: {total_val_batches}")
    print(f"Estimated time per epoch: {estimated_epoch_time / 60:.2f} minutes")
    print(
        f"Estimated total time ({num_epochs} epochs): {estimated_total_time / 3600:.2f} hours"
    )
    print("=" * 60)
    print(
        "Note: This is a rough estimate. Actual training will be slower due to model computations."
    )

    return estimated_total_time


def print_dataset_stats(dataset):
    """Print statistics about the dataset"""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")

    if hasattr(dataset, "get_category_names"):
        categories = dataset.get_category_names()
        print(f"\nNumber of categories: {len(categories)}")
        print("\nCategory names:")
        for cat_id, cat_name in sorted(categories.items()):
            print(f"  {cat_id}: {cat_name}")

    # Sample a few images to get statistics
    num_instances = []
    for i in range(min(100, len(dataset))):
        _, target = dataset[i]
        if "boxes" in target:
            num_instances.append(len(target["boxes"]))

    if num_instances:
        print(f"\nInstances per image (from {len(num_instances)} samples):")
        print(f"  Mean: {np.mean(num_instances):.2f}")
        print(f"  Median: {np.median(num_instances):.2f}")
        print(f"  Min: {min(num_instances)}")
        print(f"  Max: {max(num_instances)}")

    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Plot training history
    plot_training_history("checkpoints/training_history.json")

    # Or from a trainer object
    # plot_training_history(history_dict=trainer.history)
