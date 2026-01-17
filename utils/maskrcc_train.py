import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import os
import json


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def get_transform():
    """Simple transform to tensor"""
    return transforms.Compose([transforms.ToTensor()])


def train_model(config):
    """
    Simple training function using existing MaskRCNNTrainer

    Args:
        config: Dictionary with:
            - root_dir: Path to dataset
            - num_classes: Number of classes
            - batch_size: Batch size
            - num_epochs: Number of epochs
            - lr: Learning rate
            - checkpoint_dir: Directory for checkpoints
    """
    from datasets.isaid_dataset import iSAIDDataset
    from models.maskrcnn_model import get_maskrcnn_model, MaskRCNNTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets with filtering
    print("Loading datasets...")
    train_dataset = iSAIDDataset(
        config["root_dir"],
        split="train",
        transforms=get_transform(),
        filter_empty=True,  # Filter out empty annotations
    )

    val_dataset = iSAIDDataset(
        config["root_dir"], split="val", transforms=get_transform(), filter_empty=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    print("Creating model...")
    model = get_maskrcnn_model(config["num_classes"], pretrained=True)

    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config["lr"], momentum=0.9, weight_decay=0.0005
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("lr_step_size", 3),
        gamma=config.get("lr_gamma", 0.1),
    )

    # Create trainer (using existing MaskRCNNTrainer)
    trainer = MaskRCNNTrainer(model, optimizer, device)

    # Create checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training history
    history = {"train_loss": [], "val_loss": [], "learning_rates": []}

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch [{epoch + 1}/{config['num_epochs']}]")
        print("-" * 80)

        # Train
        train_loss = trainer.train_one_epoch(train_loader)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss = trainer.validate(val_loader)
        history["val_loss"].append(val_loss)

        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % config.get("save_every", 5) == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "history": history,
            }

            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"  ✓ Best model saved!")

        print("=" * 80)

    # Save history
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"History saved: {history_path}")

    return trainer, history


# Example usage
if __name__ == "__main__":
    config = {
        "root_dir": "iSAID_patches",
        "num_classes": 16,
        "batch_size": 2,
        "num_epochs": 20,
        "lr": 0.005,
        "lr_step_size": 5,
        "lr_gamma": 0.1,
        "num_workers": 4,
        "checkpoint_dir": "checkpoints",
        "save_every": 5,
    }

    trainer, history = train_model(config)
