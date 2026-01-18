"""
Training loop for Mask R-CNN (memory-safe version).
"""

import os
import time
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from models.maskrcnn_model import get_custom_maskrcnn
from datasets.isaid_dataset import iSAIDDataset
from training.transforms import get_transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def collate_fn(batch):
    return tuple(zip(*batch))


class Trainer:
    def __init__(
        self,
        data_root,
        num_classes=16,
        batch_size=2,
        lr=0.0001,  # Reduced from 0.005 to prevent NaN loss
        device="cuda",
        use_amp=True,
        image_size=800,  # reduced to lower VRAM
        num_workers=2,
        subset_fraction=1.0,  # Fraction of data to use (0.0 to 1.0)
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.batch_size = batch_size

        print("Loading datasets...")
        train_dataset_full = iSAIDDataset(
            data_root,
            split="train",
            transforms=get_transforms(train=True),
            image_size=image_size,
        )
        val_dataset_full = iSAIDDataset(
            data_root,
            split="val",
            transforms=get_transforms(train=False),
            image_size=image_size,
        )

        # Apply subset if needed
        if subset_fraction < 1.0:
            train_size = int(len(train_dataset_full) * subset_fraction)
            val_size = int(len(val_dataset_full) * subset_fraction)
            train_size = max(1, train_size)  # At least 1 sample
            val_size = max(1, val_size)

            self.train_dataset = Subset(train_dataset_full, range(train_size))
            self.val_dataset = Subset(val_dataset_full, range(val_size))
            print(
                f"Using {subset_fraction*100:.1f}% of data: {train_size} train, {val_size} val samples"
            )
        else:
            self.train_dataset = train_dataset_full
            self.val_dataset = val_dataset_full
            print(
                f"Using full dataset: {len(train_dataset_full)} train, {len(val_dataset_full)} val samples"
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print("Creating model...")
        self.model = get_custom_maskrcnn(num_classes=num_classes)
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

        self.scaler = GradScaler(enabled=self.use_amp)

        print(f"Device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")

        for images, targets in pbar:
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            self.optimizer.zero_grad(set_to_none=True)

            # Skip batches with no valid targets
            valid_targets = [t for t in targets if len(t["boxes"]) > 0]
            if len(valid_targets) == 0:
                continue

            try:
                if self.use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        loss_dict = self.model(images, targets)
                        loss = sum(loss_dict.values())

                    # Check for NaN/Inf before backward
                    if not torch.isfinite(loss):
                        print(f"Warning: NaN/Inf loss detected, skipping batch")
                        self.scaler.update()  # Still update scaler to prevent stall
                        del images, targets, loss, loss_dict
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients - replace NaN/Inf with zeros and clip
                    for param in self.model.parameters():
                        if param.grad is not None:
                            torch.nan_to_num_(
                                param.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())

                    # Check for NaN/Inf before backward
                    if not torch.isfinite(loss):
                        print(f"Warning: NaN/Inf loss detected, skipping batch")
                        del images, targets, loss, loss_dict
                        torch.cuda.empty_cache()
                        continue

                    loss.backward()

                    # Clip gradients - replace NaN/Inf with zeros and clip
                    for param in self.model.parameters():
                        if param.grad is not None:
                            torch.nan_to_num_(
                                param.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.optimizer.step()

                loss_value = loss.item()
                total_loss += loss_value
                # Show all losses in pbar
                loss_postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                loss_postfix["total"] = f"{loss_value:.4f}"
                pbar.set_postfix(loss=loss_postfix)
            except RuntimeError as e:
                print(f"Warning: RuntimeError in batch: {e}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            del images, targets, loss, loss_dict
            torch.cuda.empty_cache()

        self.scheduler.step()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.val_loader, desc="Validation")


        for images, targets in pbar:
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            with autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else torch.no_grad():
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict[k] for k in loss_dict)
            total_loss += loss.item()
            # Show all losses in pbar
            loss_postfix = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
            loss_postfix["total"] = f"{loss.item():.4f}"
            pbar.set_postfix(loss=loss_postfix)

            del images, targets, loss, loss_dict
            torch.cuda.empty_cache()

        return total_loss / len(self.val_loader)


    def save_checkpoint(self, path, epoch, val_loss):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )

    def fit(self, epochs=20, save_dir="checkpoints"):
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'=' * 50}")

            start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            epoch_time = time.time() - start

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            self.save_checkpoint(f"{save_dir}/last.pth", epoch, val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(f"{save_dir}/best.pth", epoch, val_loss)
                print("-> New best model saved")

            gc.collect()
            torch.cuda.empty_cache()

        print(f"\nTraining complete. Best val loss: {best_loss:.4f}")


def train(
    data_root,
    epochs=20,
    batch_size=1,
    lr=0.005,
    device="cuda",
):
    trainer = Trainer(
        data_root=data_root,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    trainer.fit(epochs=epochs)
    return trainer
