"""
Welding droplet CNN dataset utilities.

Inputs
------
sampled_frames/{index}.png
C_LABELS.csv:
index,full_x,full_y,upper_x,upper_y,lower_x,lower_y,off_x,off_y
regression_labels.csv:
index,upper_left_x,upper_left_y,upper_right_x,upper_right_y,
bottom_x,bottom_y,left_x,left_y,right_x,right_y,
full_x,full_y,upper_x,upper_y,lower_x,lower_y

Coordinates are top-left (x, y) coordinates of 32x32 crop windows,
in the ORIGINAL sampled frame coordinate system.

This script:
1. Builds classification NPZ files, splitting by source frame index.
2. Builds five regression datasets, one for each boundary point. Each
   crop contributes a sample only when that boundary point lies inside it.
"""

import cv2
import pathlib
import numpy as np

from deeplearning.settings import *
from deeplearning.utils import fileutils


# ---------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------

def get_frame(frame_path: pathlib.Path, index: int) -> np.ndarray:
    path = frame_path / f"{index}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return image


def crop_gray(frame: np.ndarray, x: int, y: int, size: int = WINDOW_SIZE) -> np.ndarray:
    """Return one grayscale crop, preserving the original pixel values."""
    height, width = frame.shape[:2]
    if x < 0 or y < 0 or x + size > width or y + size > height:
        raise ValueError(f"Crop outside frame: x={x}, y={y}, size={size}, frame_width={width}, frame_height={height}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[y:y + size, x:x + size]
    if crop.shape != (size, size):
        raise ValueError(f"Unexpected crop shape: {crop.shape}")
    return crop[..., np.newaxis]


def split_by_frame_index(
    indices: np.ndarray,
    test_ratio: float = DEFAULT_FRAME_TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split UNIQUE source-frame indices, not individual crops.
    All crops originating from one frame remain in the same split.
    """
    unique_indices = np.unique(indices)

    if len(unique_indices) < 2:
        raise ValueError("At least two distinct frame indices are required.")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_indices)

    test_count = max(1, int(round(len(shuffled) * test_ratio)))
    test_count = min(test_count, len(shuffled) - 1)

    test_indices = np.sort(shuffled[:test_count])
    train_indices = np.sort(shuffled[test_count:])
    return train_indices, test_indices


def save_npz(path: pathlib.Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# 1. Classification dataset
# ---------------------------------------------------------------------

def build_classification_npz(
    label_path: pathlib.Path = C_LABEL_PATH,
    frame_path: pathlib.Path = SAMPLE_PATH,
    output_path: pathlib.Path = DATASET_PATH / "classification_dataset.npz",
    train_frames: int = C_TRAIN_FRAMES,
    test_frames: int = C_TEST_FRAMES,
    batch_size: int = C_BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Each CSV row creates four samples in this order:
        full -> class 1
        upper -> class 2
        lower -> class 3
        off   -> class 0

    X is stored as uint8 grayscale pixels, shape (samples, 32, 32, 1).
    y is int64, shape (samples,).
    """
    rows = fileutils.CsvIO.read_from_csvfile(str(label_path))

    frame_indices: list[int] = []
    images: list[np.ndarray] = []
    labels: list[int] = []

    for row in rows:
        index = int(row["index"])
        frame = get_frame(frame_path, index)

        for category in C_LABELS:
            x = int(row[f"{category}_x"])
            y = int(row[f"{category}_y"])

            images.append(crop_gray(frame, x, y))
            labels.append(C_LABELS[category])
            frame_indices.append(index)

    X = np.stack(images).astype(np.uint8, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    source_indices = np.asarray(frame_indices, dtype=np.int64)

    unique_frames = np.unique(source_indices)
    if train_frames <= 0 or test_frames <= 0:
        raise ValueError("train_frames and test_frames must both be positive")
    if train_frames + test_frames != len(unique_frames):
        raise ValueError(
            f"Classification CSV contains {len(unique_frames)} unique frames, "
            f"but configured split requests {train_frames}+{test_frames}={train_frames + test_frames}. "
            f"Adjust CLASSIFICATION_TRAIN_FRAMES and CLASSIFICATION_TEST_FRAMES."
        )

    rng = np.random.default_rng(seed)
    shuffled_frames = rng.permutation(unique_frames)
    train_frame_indices = np.sort(shuffled_frames[:train_frames])
    test_frame_indices = np.sort(shuffled_frames[train_frames:])
    train_mask = np.isin(source_indices, train_frame_indices)
    test_mask = np.isin(source_indices, test_frame_indices)

    # Sanity checks: each frame contributes four crops and no frame leaks.
    _, crop_counts = np.unique(source_indices, return_counts=True)
    assert np.all(crop_counts == 4), "Each frame must have exactly four classification crops."
    assert set(source_indices[train_mask]).isdisjoint(set(source_indices[test_mask]))

    save_npz(
        output_path,
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        train_indices=train_frame_indices,
        test_indices=test_frame_indices,
        batch_size=np.asarray(batch_size),
        train_source_indices=source_indices[train_mask],
        test_source_indices=source_indices[test_mask],
        class_names=np.asarray(["off", "full", "upper", "lower"]),
        window_size=np.asarray(WINDOW_SIZE),
    )

    print(f"Total frames: {len(rows)}")
    print(f"Train frames: {len(train_frame_indices)}, test frames: {len(test_frame_indices)}")
    print(f"Configured classification batch_size: {batch_size}")
    print(f"X_train: {X[train_mask].shape}, y_train: {y[train_mask].shape}")
    print(f"X_test:  {X[test_mask].shape}, y_test:  {y[test_mask].shape}")


# ---------------------------------------------------------------------
# 2. Regression datasets: five targets, three possible crop windows
# ---------------------------------------------------------------------


def build_regression_npz(
    label_path: pathlib.Path = R_LABEL_PATH,
    frame_path: pathlib.Path = SAMPLE_PATH,
    output_path: pathlib.Path = DATASET_PATH / "regression_datasets.npz",
    split_mode: str = R_SPLIT_MODE,
    split_counts: dict[int, tuple[int, int]] = None,
    batch_size: int = R_BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Reconstruct the historical Dataset_reg_3_L generation logic.

    For every source frame:
      - extract full, upper, lower 32x32 grayscale crops;
      - test each of the five boundary points against each crop;
      - if a point is inside a crop, create one independent sample for
        that point's dataset (L=0..4).

    Historical label encoding:
        encoded_label = local_x + 32 * local_y
    Regression label coordinate order:
        model_label = [local_x, local_y]  # (x, y)

    Arrays saved for each target L:
      data_L_train / data_L_test / data_L_val:
          flat uint8 pixel arrays, shape (sample_count * 1024, )
          The train/test arrays remain compatible with the old next_batch().
      label_L_train / label_L_test / label_L_val:
          flat int32 arrays, shape (sample_count * 2,), each sample [x, y].
      X_L_* / y_L_*:
          convenient image and label arrays for modern training code.
      encoded_label_L_*:
          integer labels encoded as x + 32*y.
      source_index_L_* / window_name_L_*:
          source frame index and crop name for every sample.

    In sample mode, the first configured number of samples are assigned to
    training, the next configured number to testing, and all remaining
    samples to validation. This preserves the thesis train/test counts while
    retaining extra samples. Validation is an added split for the recovered
    dataset; it is not documented in thesis Table 3-3.

    In frame mode, source frames are split to avoid frame leakage; exact
    train/test counts are not guaranteed.
    """
    if split_counts is None:
        split_counts = R_SPLIT_COUNTS
    rows = fileutils.CsvIO.read_from_csvfile(str(label_path))

    # Accumulate samples separately for each boundary-point dataset.
    samples: dict[int, list[tuple[np.ndarray, int, int, str]]] = {target_id: [] for target_id in range(5)}

    for row in rows:
        index = int(row["index"])
        frame = get_frame(frame_path, index)

        windows = {name: (int(row[f"{name}_x"]), int(row[f"{name}_y"])) for name in R_WINDOWS}
        points = {name: (int(row[f"{name}_x"]), int(row[f"{name}_y"])) for name in BOUNDARY_POINTS}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for window_name, (win_x, win_y) in windows.items():
            if win_x < 0 or win_y < 0 or win_x + WINDOW_SIZE > gray.shape[1] or win_y + WINDOW_SIZE > gray.shape[0]:
                raise ValueError(f"Frame {index}: {window_name} crop outside frame: x={win_x}, y={win_y}")

            crop = gray[win_y:win_y + WINDOW_SIZE, win_x:win_x + WINDOW_SIZE]
            if crop.shape != (WINDOW_SIZE, WINDOW_SIZE):
                raise ValueError(f"Frame {index}: invalid {window_name} crop shape {crop.shape}")

            for target_id, point_name in enumerate(BOUNDARY_POINTS):
                point_x, point_y = points[point_name]
                local_x = point_x - win_x
                local_y = point_y - win_y

                # The old code checked whether the point coincided with a
                # pixel in the 32x32 crop: valid coordinates are 0..31.
                if 0 <= local_x < WINDOW_SIZE and 0 <= local_y < WINDOW_SIZE:
                    encoded_label = local_x + local_y * WINDOW_SIZE
                    samples[target_id].append((crop.copy(), index, encoded_label, window_name))

    output: dict[str, np.ndarray] = {
        "boundary_point_names": np.asarray(BOUNDARY_POINTS),
        "window_names": np.asarray(R_WINDOWS),
        "window_size": np.asarray(WINDOW_SIZE),
        "label_coordinate_order": np.asarray("x,y"),
        "label_encoding": np.asarray("x + 32*y"),
    }

    print("Regression dataset summary:")
    for target_id, point_name in enumerate(BOUNDARY_POINTS):
        target_samples = samples[target_id]
        if not target_samples:
            raise ValueError(f"No samples generated for target {target_id}: {point_name}")

        X = np.stack([item[0] for item in target_samples]).astype(np.uint8)
        source_indices = np.asarray([item[1] for item in target_samples], dtype=np.int64)
        encoded = np.asarray([item[2] for item in target_samples], dtype=np.int32)
        window_names = np.asarray([item[3] for item in target_samples])

        # Decode to the conventional [x, y] coordinate order.
        labels_xy = np.column_stack((encoded % WINDOW_SIZE, encoded // WINDOW_SIZE)).astype(np.int32)

        if split_mode == "sample":
            if target_id not in split_counts:
                raise ValueError(f"Missing split counts for regression target {target_id}")
            train_count, test_count = split_counts[target_id]
            required_count = train_count + test_count
            if len(X) < required_count:
                raise ValueError(
                    f"Regression target {target_id} ({point_name}) generated only {len(X)} samples, "
                    f"fewer than the configured train+test requirement {train_count}+{test_count}={required_count}. "
                    f"Check the CSV and crop annotations."
                )

            # Preserve the configured thesis counts. Any additional samples
            # are retained as a separate validation set, not merged into test.
            train_mask = np.zeros(len(X), dtype=bool)
            test_mask = np.zeros(len(X), dtype=bool)
            val_mask = np.zeros(len(X), dtype=bool)
            train_mask[:train_count] = True
            test_mask[train_count:required_count] = True
            val_mask[required_count:] = True
            train_frames = np.unique(source_indices[train_mask])
            test_frames = np.unique(source_indices[test_mask])
            val_frames = np.unique(source_indices[val_mask])
        elif split_mode == "frame":
            train_frames, test_frames = split_by_frame_index(
                source_indices, test_ratio=(
                    split_counts[target_id][1] / sum(split_counts[target_id])
                    if target_id in split_counts else DEFAULT_FRAME_TEST_RATIO
                ), seed=seed + target_id
            )
            train_mask = np.isin(source_indices, train_frames)
            test_mask = np.isin(source_indices, test_frames)
            val_mask = np.zeros(len(X), dtype=bool)
            val_frames = np.asarray([], dtype=np.int64)
            if not set(source_indices[train_mask]).isdisjoint(set(source_indices[test_mask])):
                raise AssertionError("Source-frame leakage detected")
        else:
            raise ValueError("split_mode must be 'sample' or 'frame'")

        # Old next_batch() expects flat data and label vectors.
        for split_name, mask in (("train", train_mask), ("test", test_mask), ("val", val_mask)):
            output[f"data_{target_id}_{split_name}"] = X[mask].reshape(-1)
            output[f"label_{target_id}_{split_name}"] = labels_xy[mask].reshape(-1)
            output[f"X_{target_id}_{split_name}"] = X[mask][..., np.newaxis]
            output[f"y_{target_id}_{split_name}"] = labels_xy[mask]
            output[f"encoded_label_{target_id}_{split_name}"] = encoded[mask]
            output[f"source_index_{target_id}_{split_name}"] = source_indices[mask]
            output[f"window_name_{target_id}_{split_name}"] = window_names[mask]

        output[f"train_frame_indices_{target_id}"] = train_frames
        output[f"test_frame_indices_{target_id}"] = test_frames
        output[f"val_frame_indices_{target_id}"] = val_frames
        output[f"batch_size_{target_id}"] = np.asarray(batch_size)
        output[f"configured_train_count_{target_id}"] = np.asarray(int(np.sum(train_mask)))
        output[f"configured_test_count_{target_id}"] = np.asarray(int(np.sum(test_mask)))
        output[f"generated_val_count_{target_id}"] = np.asarray(int(np.sum(val_mask)))

        actual_train_count = int(np.sum(train_mask))
        actual_test_count = int(np.sum(test_mask))
        actual_val_count = int(np.sum(val_mask))
        window_distribution = dict(zip(*np.unique(window_names, return_counts=True)))
        print(
            f"  {target_id} {point_name}: total={len(X)}, "
            f"train={actual_train_count}, test={actual_test_count}, "
            f"validation={actual_val_count}, windows={window_distribution}"
        )

    save_npz(output_path, **output)
    print("Saved five target-specific regression datasets in one NPZ file.")
    print("Labels are stored in coordinate order [x, y].")
    if split_mode == "sample":
        print("Extra samples are retained as validation data (not part of thesis Table 3-3).")
    print(f"Regression split mode: {split_mode}")
    print(f"Configured regression batch_size: {batch_size}")


if __name__ == "__main__":
    # Build the classification dataset NPZ:
    # build_classification_npz()

    # Build five target-specific regression datasets:
    # build_regression_npz()
    pass
