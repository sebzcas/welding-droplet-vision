import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from deeplearning.models.classifier import ClassificationCNN
from deeplearning.models.regressor import RegressionCNN
from deeplearning.settings import *
from deeplearning.utils import fileutils


# ============================================================
# Detection policy
# ============================================================
TARGET_POLICIES = {
    0: {
        "name": "upper_left",
        "cluster_distance_sq": 2,
        "selection": "max_y_point",
    },
    1: {
        "name": "upper_right",
        "cluster_distance_sq": 2,
        "selection": "min_y_point",
    },
    2: {
        "name": "bottom",
        "cluster_distance_sq": None,
        "selection": "max_y_all",
    },
    3: {
        "name": "left",
        "cluster_distance_sq": 2,
        "selection": "min_x_max_y",
    },
    4: {
        "name": "right",
        "cluster_distance_sq": 8,
        "selection": "max_x_max_y",
    },
}


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(path, model, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model


def load_all_models(args, device):
    print("\nLoading classification model...")
    classifier = ClassificationCNN()
    classifier = load_checkpoint(Path(args.classifier_dir) / "classification_cnn.pt", classifier, device)
    print("Classification model loaded.")

    regressors = dict()
    for target_id, target_name in enumerate(BOUNDARY_POINTS):
        checkpoint_path = (Path(args.regression_dir) / f"target_{target_id}_{target_name}" / "regression_cnn.pt")
        print(f"Loading regression model {target_id}: {target_name}")
        model = RegressionCNN()
        model = load_checkpoint(checkpoint_path, model, device,)
        regressors[target_id] = model
    print("All five regression models loaded.\n")

    return classifier, regressors


# ============================================================
# Batched inference
# ============================================================

@torch.inference_mode()
def predict_classification(model, crops, device, batch_size=100):
    """
    Returns:
        labels: (N,)
        probabilities: (N, 4)
    """

    all_labels = []
    all_probabilities = []

    for start in range(0, len(crops), batch_size):
        end = min(start + batch_size, len(crops))
        x = torch.from_numpy(np.ascontiguousarray(crops[start:end])).to(device)
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_probabilities)


@torch.inference_mode()
def predict_regression(model, crops, device, batch_size=80):
    """
    Returns:
        predictions: (N, 2), coordinate order [x, y]
    """
    predictions = []
    for start in range(0, len(crops), batch_size):
        end = min(start + batch_size, len(crops))
        x = torch.from_numpy(np.ascontiguousarray(crops[start:end])).to(device)
        output = model(x)
        predictions.append(output.cpu().numpy())

    if not predictions:
        return np.empty((0, 2), dtype=np.float32)

    return np.concatenate(predictions, axis=0)


# ============================================================
# Spatial clustering
# ============================================================

def find_clusters(points, distance_sq_threshold):
    """
    Find connected components using squared Euclidean distance.
    points: ndarray (N, 2), [x, y]
    """
    n = len(points)
    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True

        stack = [start]
        cluster_indices = []
        while stack:
            current = stack.pop()
            cluster_indices.append(current)
            diff = points - points[current]
            distances_sq = np.sum(diff ** 2, axis=1)
            neighbors = np.where((distances_sq <= distance_sq_threshold) & (~visited))[0]
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
        clusters.append(np.asarray(cluster_indices, dtype=int))

    return clusters


def select_representative(points, policy):
    """
    Apply the original target-specific selection strategy.

    points: ndarray (N, 2), [x, y]

    Returns:
        point: tuple (x, y) or None
        cluster_size: int
        cluster_predictions: ndarray (M, 2)
    """

    if len(points) == 0:
        return None, 0, np.empty((0, 2), dtype=np.float32)

    selection = policy["selection"]

    # Bottom: select the lowest point among ALL predictions.
    if selection == "max_y_all":
        index = np.argmax(points[:, 1])
        return tuple(points[index].astype(float)), 1, points[[index]]

    # Other four targets: select the largest connected component.
    clusters = find_clusters(points, policy["cluster_distance_sq"])

    largest_cluster_indices = max(clusters, key=len)

    cluster = points[largest_cluster_indices]

    if selection == "max_y_point":
        point = cluster[np.argmax(cluster[:, 1])]

    elif selection == "min_y_point":
        point = cluster[np.argmin(cluster[:, 1])]

    elif selection == "min_x_max_y":
        point = np.array([np.min(cluster[:, 0]), np.max(cluster[:, 1])])

    elif selection == "max_x_max_y":
        point = np.array([np.max(cluster[:, 0]), np.max(cluster[:, 1]),])

    else:
        raise ValueError(f"Unknown selection policy: {selection}")

    return tuple(point.astype(float)), len(cluster), cluster


# ============================================================
# Complete detection
# ============================================================

def generate_windows(image, margin=15, stride=5, classifier=None, device=None, cls_batch_size=100, down_scan_steps=3):
    """
    Generate grayscale windows row by row.

    If classifier is provided, stop scanning after the first row
    containing a "full" prediction plus down_scan_steps additional rows.

    Returns:
        crops: (N, 1, 32, 32), float32, normalized to [0, 1]
        origins: (N, 2), [window_x, window_y]
        labels: (N,), classification labels, or None if classifier is None
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    crops = []
    origins = []
    all_labels = []

    first_full_row = None
    row_index = 0

    for y in range(margin, height - WINDOW_SIZE - margin, stride):
        row_crops = []
        row_origins = []

        for x in range(margin, width - WINDOW_SIZE - margin, stride):
            crop = gray[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

            if crop.shape != (WINDOW_SIZE, WINDOW_SIZE):
                continue

            crop = crop.astype(np.float32) / 255.0

            row_crops.append(crop)
            row_origins.append([x, y])

        if not row_crops:
            continue

        row_crops = np.asarray(row_crops, dtype=np.float32)[:, None, :, :]
        row_origins = np.asarray(row_origins, dtype=np.int32)

        # Classify the current row.
        if classifier is not None:
            row_labels, _ = predict_classification(classifier, row_crops, device, batch_size=cls_batch_size)
            all_labels.extend(row_labels.tolist())

            # Record the first row containing a "full" window.
            if first_full_row is None and np.any(row_labels == 1):
                first_full_row = row_index

        crops.append(row_crops)
        origins.append(row_origins)

        # Stop after scanning the specified number of additional rows.
        if (
            first_full_row is not None
            and row_index >= first_full_row + down_scan_steps
        ):
            break

        row_index += 1

    if not crops:
        raise ValueError(
            "No sliding windows generated. Check image size, margin and stride."
        )

    crops = np.concatenate(crops, axis=0)
    origins = np.concatenate(origins, axis=0)

    labels = (
        np.asarray(all_labels, dtype=np.int64)
        if classifier is not None
        else None
    )

    return crops, origins, labels


def detect(image, classifier, regressors, device, args, verbose=False):
    """
    Detect all five droplet limits.
    """

    # --------------------------------------------------------
    # 1. Generate windows and classify row by row
    # --------------------------------------------------------
    crops, origins, labels = generate_windows(
        image,
        margin=args.margin,
        stride=args.stride,
        classifier=classifier,
        device=device,
        cls_batch_size=args.cls_batch_size,
        down_scan_steps=args.down_scan_steps,
    )

    candidate_indices = np.where(labels == 1)[0]

    if verbose:
        tqdm.write(f"Generated sliding windows: {len(crops)}")
        tqdm.write(f"Classification results: {len(candidate_indices)} full windows")
        for class_name, class_id in C_LABELS.items():
            count = int(np.sum(labels == class_id))
            tqdm.write(f"  {class_name:>5}: {count}")

    # Coordinates of windows classified as "full"
    full_windows = origins[candidate_indices]

    detection = {
        "full_windows": full_windows,
        "candidate_indices": candidate_indices,
        "results": {},
    }

    if len(candidate_indices) == 0:
        print("Warning: no windows classified as 'full'.")

        for target_id in range(len(BOUNDARY_POINTS)):
            detection["results"][target_id] = {
                "point": None,
                "cluster_size": 0,
                "candidate_count": 0,
                "predictions": np.empty((0, 2), dtype=np.float32),
                "cluster_predictions": np.empty((0, 2), dtype=np.float32),
            }

        return detection

    # Only retain full windows
    candidate_crops = crops[candidate_indices]
    candidate_origins = origins[candidate_indices]

    # --------------------------------------------------------
    # 2. Five independent regression models
    # --------------------------------------------------------
    for target_id, target_name in enumerate(BOUNDARY_POINTS):
        if verbose:
            tqdm.write(f"Detecting {target_name}...")

        model = regressors[target_id]

        local_predictions = predict_regression(
            model,
            candidate_crops,
            device,
            batch_size=args.reg_batch_size,
        )

        # ----------------------------------------------------
        # 3. Convert local [x, y] to global [x, y]
        # ----------------------------------------------------
        global_predictions = local_predictions + candidate_origins

        # Keep predictions inside the original image
        valid_mask = (
            (global_predictions[:, 0] >= 0)
            & (global_predictions[:, 0] < image.shape[1])
            & (global_predictions[:, 1] >= 0)
            & (global_predictions[:, 1] < image.shape[0])
        )

        global_predictions = global_predictions[valid_mask]

        # ----------------------------------------------------
        # 4. Spatial clustering
        # ----------------------------------------------------
        policy = TARGET_POLICIES[target_id]

        representative, cluster_size, cluster_predictions = select_representative(global_predictions, policy)
        if representative is None:
            if verbose:
                tqdm.write("  No valid predictions.")
            detection["results"][target_id] = {
                "point": None,
                "cluster_size": 0,
                "candidate_count": len(global_predictions),
                "predictions": global_predictions,
                "cluster_predictions": np.empty((0, 2), dtype=np.float32),
            }
            continue

        point = np.rint(representative).astype(int)
        px, py = int(point[0]), int(point[1])

        detection["results"][target_id] = {
            "point": (px, py),
            "cluster_size": cluster_size,
            "candidate_count": len(global_predictions),
            "predictions": global_predictions,
            "cluster_predictions": cluster_predictions,
        }

        if verbose:
            tqdm.write(f"  Candidate predictions: {len(global_predictions)}")
            tqdm.write(f"  Largest cluster size: {cluster_size}")
            tqdm.write(f"  Final coordinate [x, y]: ({px}, {py})")

    return detection


def draw_annotations(image, detection):
    """
    Draw classification windows and regression predictions.

    Args:
        image: Original image.
        detection: Result returned by detect().

    Returns:
        Annotated image.
    """
    preview = cv2.resize(image.copy(), None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)

    # --------------------------------------------------------
    # 1. Draw windows classified as "full"
    # --------------------------------------------------------
    window_size = WINDOW_SIZE * ZOOM_FACTOR
    for x, y in detection["full_windows"]:
        px = int(x * ZOOM_FACTOR)
        py = int(y * ZOOM_FACTOR)
        cv2.rectangle(preview, (px, py), (px + window_size, py + window_size), (0, 255, 0), 1)

    # --------------------------------------------------------
    # 2. Draw regression predictions and final points
    # --------------------------------------------------------
    for target_id, target_name in enumerate(BOUNDARY_POINTS):
        result = detection["results"][target_id]
        color = TARGET_COLORS[target_id]

        # Draw all valid regression predictions
        for x, y in result["predictions"]:
            px = int(round(x * ZOOM_FACTOR))
            py = int(round(y * ZOOM_FACTOR))
            cv2.circle(preview, (px, py), 2, color, -1)

        # Draw the final representative point
        if result["point"] is not None:
            x, y = result["point"]
            px = int(x * ZOOM_FACTOR)
            py = int(y * ZOOM_FACTOR)
            cv2.drawMarker(preview, (px, py), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(preview, target_name, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return preview


def get_frame_paths(video_dir):
    """
    Get all image frames in a video directory.

    Frame filenames must have numeric stems, e.g.:
        0.png
        1.png
        100.png
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    frame_paths = [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in image_extensions]

    indexed_paths = []

    for path in frame_paths:
        try:
            frame_index = int(path.stem)
        except ValueError:
            raise ValueError(f"Frame filename must be numeric: {path.name}")
        indexed_paths.append((frame_index, path))

    indexed_paths.sort(key=lambda item: item[0])

    # Avoid overwriting frames with duplicate indices.
    indices = [item[0] for item in indexed_paths]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate frame indices found in {video_dir}")

    return indexed_paths


def process_video(video_name, classifier, regressors, device, args):
    """
    Process all frames belonging to one video.

    Input:
        DATA_PATH / "frames" / video_name

    Outputs:
        DATA_PATH / "outputs" / video_name / frame_index.png
        DATA_PATH / "outputs" / video_name.csv
    """

    input_dir = FRAME_PATH / video_name
    output_dir = OUTPUTS_PATH / video_name
    csv_path = OUTPUTS_PATH / f"{video_name}.csv"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Video frame directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = get_frame_paths(input_dir)
    if not frame_paths:
        print(f"Warning: no frames found in {input_dir}")
        return

    print("\n" + "=" * 65)
    print(f"Processing video: {video_name}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total frames:     {len(frame_paths)}")
    print("=" * 65)

    records = []

    progress_bar = tqdm(frame_paths, desc=f"Processing {video_name}", unit="frame", dynamic_ncols=True,)

    for frame_number, (frame_index, frame_path) in enumerate(progress_bar, start=1):
        progress_bar.set_postfix_str(f"Frame {frame_index}")
        # print(f"\n[{frame_number}/{len(frame_paths)}] Processing frame {frame_index}: {frame_path.name}")

        # ------------------------------------------------
        # 1. Read original frame
        # ------------------------------------------------
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")

        # ------------------------------------------------
        # 2. Detect five droplet limits
        # ------------------------------------------------
        detection = detect(image, classifier, regressors, device, args, verbose=False)

        # ------------------------------------------------
        # 3. Draw annotations
        # ------------------------------------------------
        annotated = draw_annotations(image, detection)

        # ------------------------------------------------
        # 4. Save annotated image
        # ------------------------------------------------
        output_path = output_dir / f"{frame_index}.png"

        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to save image: {output_path}")

        # ------------------------------------------------
        # 5. Collect coordinates
        # ------------------------------------------------
        row = {"frame_index": frame_index}

        for target_id, target_name in enumerate(BOUNDARY_POINTS):
            point = detection["results"][target_id]["point"]

            if point is None:
                row[f'{target_name}_x'] = None
                row[f'{target_name}_y'] = None
            else:
                x, y = point
                row[f'{target_name}_x'] = x
                row[f'{target_name}_y'] = y

        records.append(row)

        progress_bar.set_postfix(frame=frame_index, bottom=detection["results"][2]["point"])
        # print(f"Saved image: {output_path}")
        # print(f"Coordinates: {row}")

    fileutils.CsvIO.write_to_csvfile(str(csv_path), records)
    print("\n" + "-" * 65)
    print(f"Video processing completed: {video_name}")
    print(f"Annotated frames: {output_dir}")
    print(f"Coordinate CSV:   {csv_path}")
    print("-" * 65)


def detect_video(video_names: list[str] = None):
    parser = argparse.ArgumentParser(description="Welding droplet five-limit detection")
    parser.add_argument("input", nargs="?", default=video_names, help="Input image path or video name")
    parser.add_argument(
        "--classifier-dir", type=str, default=Path(BASE_PATH) / "deeplearning" / "training" / "runs" / "classification")
    parser.add_argument(
        "--regression-dir", type=str, default=Path(BASE_PATH) / "deeplearning" / "training" / "runs" / "regression")
    parser.add_argument("--margin", type=int, default=15)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--down-scan-steps", type=int, default=3, help="Additional rows to scan after the first full detection.")
    parser.add_argument("--cls-batch-size", type=int, default=100)
    parser.add_argument("--reg-batch-size", type=int, default=80)
    parser.add_argument("--cluster-distance", type=float, default=2.0, help="Euclidean distance threshold in pixels")
    parser.add_argument("--output", type=str, default=None, help="Output annotated image path")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    allowed_video_names = [video_info["name"] for video_info in VIDEOS_INFO]

    # --------------------------------------------------------
    # 1. Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --------------------------------------------------------
    # 2. Load models
    # --------------------------------------------------------
    classifier, regressors = load_all_models(args, device)

    # --------------------------------------------------------
    # 3. Determine processing mode
    # --------------------------------------------------------
    if isinstance(args.input, str) and Path(args.input).suffix.lower() == ".png":
        # ====================================================
        # Single-image detection
        # ====================================================
        image_path = Path(args.input)

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        print(f"\nImage size: {image.shape[1]} x {image.shape[0]}")

        detection = detect(image, classifier, regressors, device, args)

        # Print summary
        print("\n" + "=" * 55)
        print("FINAL DETECTION RESULTS")
        print("=" * 55)

        for target_id, target_name in enumerate(BOUNDARY_POINTS):
            result = detection["results"][target_id]

            if result["point"] is None:
                print(f"{target_id} {target_name:>12}: Not detected")
            else:
                x, y = result["point"]
                print(f"{target_id} {target_name:>12}: x={x:4d}, y={y:4d} (cluster={result['cluster_size']})")

        print("=" * 55)

        # Draw annotations
        annotated = draw_annotations(image, detection)

        # Save result
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = OUTPUTS_PATH / f"{image_path.stem}_detected.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Failed to save image: {output_path}")
        print(f"\nAnnotated image saved to: {output_path}")

        # Display
        if not args.no_show:
            cv2.imshow("Droplet Five-Limit Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # ====================================================
        # Batch video detection
        # ====================================================
        frames_root = FRAME_PATH

        if not frames_root.is_dir():
            raise FileNotFoundError(f"Frames directory not found: {frames_root}")

        video_names = []
        if isinstance(args.input, str):
            if (video_name := Path(args.input).name) in allowed_video_names:
                video_names = [video_name]
            elif video_name == "*":
                video_names = allowed_video_names
        else:
            video_names = args.input
        if not video_names:
            print(f"Input a image, a video-name in {allowed_video_names}, or *.")

        print(f"\nVideos to process:")
        for video_name in video_names:
            print(f"- Video Name: {video_name}")
            process_video(video_name, classifier, regressors, device, args)

        print("\n" + "=" * 65)
        print("ALL VIDEO PROCESSING COMPLETED")
        print("=" * 65)


if __name__ == "__main__":
    videoNames = ["20", "21", "22", "24c", "26c", "29", "33"]
    detect_video()
