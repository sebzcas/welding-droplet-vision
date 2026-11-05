"""
PyTorch reproduction of the original TensorFlow classification CNN.

NPZ schema:
    X_train, y_train, X_test, y_test
Images: grayscale 32x32, uint8 [0, 255] (also accepts float [0,255] or [0,1]).
Labels: off=0, full=1, upper=2, lower=3.

Run:
    python train_classification.py --npz classification_dataset.npz
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader, TensorDataset

from deeplearning.models.classifier import ClassificationCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_images(array: np.ndarray) -> torch.Tensor:
    """Convert (N,32,32), (N,32,32,1), or (N,1,32,32) to float NCHW."""
    x = np.asarray(array)

    if x.ndim == 3:
        # N,H,W
        x = x[:, None, :, :]
    elif x.ndim == 4:
        if x.shape[-1] == 1:       # NHWC
            x = np.transpose(x, (0, 3, 1, 2))
        elif x.shape[1] == 1:      # NCHW
            pass
        else:
            raise ValueError(f"Unsupported image shape: {x.shape}")
    else:
        raise ValueError(f"Expected 3D/4D image array, got {x.shape}")

    if x.shape[1:] != (1, 32, 32):
        raise ValueError(f"Expected 32x32 grayscale images, got {x.shape}")

    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    # The generated NPZ stores grayscale pixel values in [0,255].
    if x.numel() and x.max().item() > 1.0:
        x = x / 255.0
    return x


def load_dataset(npz_path: Path):
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path) as z:
        required = ("X_train", "y_train", "X_test", "y_test")
        if missing := [key for key in required if key not in z]:
            raise KeyError(f"Missing NPZ keys: {missing}. Available keys: {list(z.keys())}")

        X_train = prepare_images(z["X_train"])
        y_train = torch.as_tensor(np.asarray(z["y_train"]).reshape(-1), dtype=torch.long)
        X_test = prepare_images(z["X_test"])
        y_test = torch.as_tensor(np.asarray(z["y_test"]).reshape(-1), dtype=torch.long)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Image and label counts do not match.")
    if y_train.numel() and (y_train.min() < 0 or y_train.max() > 3):
        raise ValueError("Classification labels must be in [0,3].")

    return (
        TensorDataset(X_train, y_train),
        TensorDataset(X_test, y_test),
    )


def classification_loss(logits, labels, model):
    cross_entropy = nn_functional.cross_entropy(logits, labels)
    # TF used 0.004 * tf.nn.l2_loss(W) for fc1 and fc2.
    # tf.nn.l2_loss(W) = 0.5 * sum(W**2), hence coefficient 0.002 here.
    l2_penalty = 0.002 * (model.fc1.weight.square().sum() + model.fc2.weight.square().sum())
    return cross_entropy + l2_penalty


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = nn_functional.cross_entropy(logits, labels)

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_n

    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path("classification_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/classification"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set, test_set = load_dataset(args.npz)
    print(f"Train samples: {len(train_set)}")
    print(f"Test samples:  {len(test_set)}")

    # Original code traversed the prepared training set in batches.
    # drop_last=True preserves its full-batch training behavior.
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

    if len(train_loader) == 0:
        raise ValueError("No full training batch. Reduce --batch-size.")

    model = ClassificationCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy", "seconds"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            start = time.time()

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = classification_loss(logits, labels, model)
                loss.backward()
                optimizer.step()

                n = labels.size(0)
                running_loss += loss.item() * n
                seen += n

            train_loss = running_loss / seen
            test_loss, test_accuracy = evaluate(model, test_loader, device)
            elapsed = time.time() - start

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"test_loss={test_loss:.6f} | "
                f"test_accuracy={test_accuracy:.4%} | "
                f"time={elapsed:.1f}s"
            )
            writer.writerow([epoch, train_loss, test_loss, test_accuracy, elapsed])
            f.flush()

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": ["off", "full", "upper", "lower"],
                    "args": vars(args),
                },
                args.output / "classification_cnn.pt"
            )

    print(f"Model saved to: {args.output / 'classification_cnn.pt'}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
