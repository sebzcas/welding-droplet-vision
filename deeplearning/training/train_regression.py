"""
PyTorch reproduction of the original TensorFlow regression CNN.

Expected NPZ keys for target L (0..4):
    data_L_train, label_L_train
    data_L_test,  label_L_test
    data_L_val,   label_L_val       (optional)

The data arrays contain 32x32 grayscale crops, usually flattened to 1024 values.
The labels are [y, x], matching the original TensorFlow reader/model output order.

Target mapping:
    0: upper-left limit
    1: upper-right limit
    2: bottom limit
    3: left limit
    4: right limit

Run one target:
    python train_regression.py --npz regression_dataset.npz --target 0

Run another target by changing --target to 1, 2, 3, or 4.
"""

import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader, TensorDataset

from deeplearning.models.regressor import RegressionCNN
from deeplearning.settings import *


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_images(array: np.ndarray) -> torch.Tensor:
    """Convert flattened or image-shaped crops to float NCHW."""
    x = np.asarray(array)

    # Flat uint8 array: (N * 1024,)
    if x.ndim == 1:
        if x.size % 1024 != 0:
            raise ValueError(f"Flattened image array size must be divisible by 1024, got {x.size}")
        x = x.reshape(-1, 1, 32, 32)

    elif x.ndim == 2:
        if x.shape[1] == 1024:
            x = x.reshape(-1, 1, 32, 32)
        elif x.shape[1] == 32 and x.shape[0] % 32 == 0:
            x = x.reshape(-1, 1, 32, 32)
        else:
            raise ValueError(f"Unsupported image shape: {x.shape}")

    elif x.ndim == 3 and x.shape[1:] == (32, 32):
        x = x[:, None, :, :]

    elif x.ndim == 4:
        if x.shape[1:] == (1, 32, 32):
            pass
        elif x.shape[1:] == (32, 32, 1):
            x = np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unsupported image shape: {x.shape}")

    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    if x.shape[1:] != (1, 32, 32):
        raise ValueError(f"Expected 32x32 grayscale images, got {x.shape}")

    x = torch.from_numpy(np.ascontiguousarray(x)).float()

    if x.numel() and x.max().item() > 1.0:
        x = x / 255.0

    return x


def prepare_labels(array: np.ndarray) -> torch.Tensor:
    """Prepare regression labels in [x, y] order."""
    y = np.asarray(array)

    if y.ndim == 1:
        if y.size % 2 != 0:
            raise ValueError(f"Flattened label array size must be divisible by 2, got {y.size}")
        y = y.reshape(-1, 2)

    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected labels with shape (N,2), got {y.shape}")

    return torch.from_numpy(np.ascontiguousarray(y)).float()


def load_split(z, target: int, split: str):
    data_key = f"data_{target}_{split}"
    label_key = f"label_{target}_{split}"

    if data_key not in z or label_key not in z:
        if split == "val":
            return None
        raise KeyError(f"Missing keys {data_key!r}/{label_key!r}. Available keys: {list(z.keys())}")

    X = prepare_images(z[data_key])
    y = prepare_labels(z[label_key])

    if len(X) != len(y):
        raise ValueError(f"{data_key} and {label_key} sample counts differ.")
    if y.numel() and (y.min() < 0 or y.max() > 31):
        raise ValueError(f"{label_key} contains coordinates outside [0,31]. Check the [x,y] labels and NPZ generation.")
    return TensorDataset(X, y)


def original_regression_accuracy(prediction, target):
    # TensorFlow: mean(1 - abs((logits-labels)/32)).
    return (1.0 - torch.abs((prediction - target) / 32.0)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_original_accuracy = 0.0
    total_within_2px = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)
        error = prediction - labels
        n = labels.size(0)

        total_squared_error += nn_functional.mse_loss(prediction, labels, reduction="sum").item()
        total_absolute_error += error.abs().sum().item()
        total_original_accuracy += (original_regression_accuracy(prediction, labels).item() * n)

        # Reproduce the original code's int32 cast before checking > 2.
        # Conversion truncates toward zero, as TensorFlow's cast did.
        delta_int = error.to(torch.int32)
        bad_sample = (delta_int.abs() > 2).any(dim=1)
        total_within_2px += (~bad_sample).sum().item()
        total_samples += n

    coordinate_count = total_samples * 2
    return {
        "mse": total_squared_error / coordinate_count,
        "mae": total_absolute_error / coordinate_count,
        "original_accuracy": total_original_accuracy / total_samples,
        "within_2px_sample_accuracy": total_within_2px / total_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("regression_dataset.npz"))
    parser.add_argument("--target", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/regression"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.npz.is_file():
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")

    with np.load(args.npz) as z:
        train_set = load_split(z, args.target, "train")
        test_set = load_split(z, args.target, "test")
        val_set = load_split(z, args.target, "val")

    print(f"Target: {args.target} ({BOUNDARY_POINTS[args.target]})")
    print(f"Train samples: {len(train_set)}")
    print(f"Test samples:  {len(test_set)}")
    print(f"Validation samples: {len(val_set) if val_set is not None else 0}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        drop_last=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = (
        DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )
        if val_set is not None else None
    )

    if len(train_loader) == 0:
        raise ValueError("No full training batch. Reduce --batch-size.")

    model = RegressionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    target_dir = args.output / f"target_{args.target}_{BOUNDARY_POINTS[args.target]}"
    target_dir.mkdir(parents=True, exist_ok=True)
    history_path = target_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_mse", "monitor_mse", "monitor_mae",
            "monitor_original_accuracy", "monitor_within_2px_accuracy", "seconds"
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_squared_error = 0.0
            seen_coordinates = 0
            start = time.time()

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                prediction = model(images)

                # Original regression loss: mean over batch and both coordinates.
                loss = nn_functional.mse_loss(prediction, labels)
                loss.backward()
                optimizer.step()

                n_coordinates = labels.numel()
                running_squared_error += loss.item() * n_coordinates
                seen_coordinates += n_coordinates

            train_mse = running_squared_error / seen_coordinates

            # Use the extra validation partition for monitoring when present.
            monitor_loader = val_loader if val_loader is not None else test_loader
            monitor = evaluate(model, monitor_loader, device)
            elapsed = time.time() - start

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_mse={train_mse:.6f} | "
                f"monitor_mse={monitor['mse']:.6f} | "
                f"monitor_mae={monitor['mae']:.4f}px | "
                f"original_accuracy={monitor['original_accuracy']:.4%} | "
                f"within_2px={monitor['within_2px_sample_accuracy']:.4%} | "
                f"time={elapsed:.1f}s"
            )

            writer.writerow([
                epoch, train_mse, monitor["mse"], monitor["mae"],
                monitor["original_accuracy"],
                monitor["within_2px_sample_accuracy"], elapsed
            ])
            f.flush()

            torch.save(
                {
                    "target": args.target,
                    "target_name": BOUNDARY_POINTS[args.target],
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "label_order": ["x", "y"],
                    "args": vars(args),
                },
                target_dir / "regression_cnn.pt"
            )

    # Final evaluation on the held-out test set.
    test_metrics = evaluate(model, test_loader, device)
    print("\nFinal test metrics:")
    print(f"  MSE (per coordinate):       {test_metrics['mse']:.6f}")
    print(f"  MAE (per coordinate):       {test_metrics['mae']:.4f} pixels")
    print(f"  Original accuracy metric:   {test_metrics['original_accuracy']:.4%}")
    print(f"  Within-2px sample accuracy: {test_metrics['within_2px_sample_accuracy']:.4%}")

    print(f"\nModel saved to: {target_dir / 'regression_cnn.pt'}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
