
import cv2
import random
import pathlib

from deeplearning.settings import *
from deeplearning.utils import fileutils


# ==================== Window Sampling ====================

def sample_positive_windows(x: int, y: int, width: int, height: int, size: int = WINDOW_SIZE) -> list[tuple[int, int]]:
    windows = [
        (x - size // 2, y - size // 2),
        (random.randint(x - int(0.75 * size), x - int(0.25 * size)), random.randint(y - size, y - int(0.75 * size))),
        (random.randint(x - int(0.75 * size), x - int(0.25 * size)), random.randint(y - int(0.25 * size), y))]
    return [(max(0, min(wx, width - size)), max(0, min(wy, height - size))) for wx, wy in windows]


def windows_overlap(a: tuple[int, int], b: tuple[int, int], size: int) -> bool:
    """Check whether two square windows overlap."""
    ax, ay = a
    bx, by = b
    return ax < bx + size and ax + size > bx and ay < by + size and ay + size > by


def sample_negative_window(
        positive_windows: list[tuple[int, int]], width: int, height: int,
        size: int = WINDOW_SIZE, max_attempts: int = 10000
) -> tuple[int, int]:
    full, upper, lower = positive_windows

    # Negative window center must not be lower than the lower window center.
    max_y = min(height - size, lower[1])

    # Upper window vertical midpoint.
    upper_mid_y = upper[1] + size / 2

    for _ in range(max_attempts):
        negative = (random.randint(0, width - size), random.randint(0, max_y))
        nx, ny = negative

        # Negative window must not overlap full or lower.
        if windows_overlap(negative, full, size):
            continue

        if windows_overlap(negative, lower, size):
            continue

        # Overlap with upper is allowed if the entire negative window is above its midpoint.
        if windows_overlap(negative, upper, size) and ny + size > upper_mid_y:
            continue

        return negative

    raise RuntimeError(f"Cannot find a valid negative window after {max_attempts} attempts.")


# ==================== Label Utilities ====================

def load_existing_indices(label_path: pathlib.Path) -> set[int]:
    """Read frame indices already saved."""
    if not label_path.exists():
        return set()
    with label_path.open("r", encoding="utf-8") as f:
        records = fileutils.CsvIO.to_records(f.read())
    return {int(x["index"]) for x in records}


def write_data(label_path: pathlib.Path, frame_index: int, windows: list[tuple[int, int]]):
    """Save four crop window coordinates."""
    if len(windows) != 4:
        raise ValueError("Exactly four windows are required.")

    coordinates_text = ",".join(f"{x},{y}" for x, y in windows)

    record = f'{frame_index},{coordinates_text}\n'

    content = ""
    if label_path.exists():
        with open(label_path, 'r', encoding="utf-8") as f:
            content = f.read()

    with open(label_path, 'a', encoding="utf-8") as f:
        if not content:
            f.write(",".join((
                "index",
                "full_x", "full_y",
                "upper_x", "upper_y",
                "lower_x", "lower_y",
                "off_x", "off_y",
            )) + "\n")
        f.write(record)

    print(f"Saved: {record.strip()}")


class ClassificationAnnotator:
    def __init__(self, start_index: int = None):
        self.frame_paths = sorted(SAMPLE_PATH.glob("*.png"), key=lambda path: int(path.stem))
        if not self.frame_paths:
            raise FileNotFoundError(f"No PNG images found in {SAMPLE_PATH}")

        # Locate the starting frame.

        self.existing_indices = load_existing_indices(C_LABEL_PATH)
        if start_index is None:
            self.frame_index = (list(self.existing_indices)[-1] + 1) if self.existing_indices else 0
        else:
            self.frame_index = start_index

        self.original_image = None
        self.display_image = None

        # Current frame annotation.
        self.center = None
        self.windows = []

        DATASET_PATH.mkdir(parents=True, exist_ok=True)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouse_callback)

        self.load_frame()

    def load_frame(self):
        if self.frame_index >= len(self.frame_paths):
            print("Dataset generation completed.")
            return

        image_file = SAMPLE_PATH / f"{self.frame_index}.png"
        image = cv2.imread(str(image_file))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_file}")

        self.original_image = image
        self.display_image = cv2.resize(image, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)

        self.center = None
        self.windows = []

        print(f"\nCurrent frame: {self.frame_index}")

    def mouse_callback(self, event, x, y, flags, param):
        if self.original_image is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.center is not None:
                print("Right-click to cancel the current selection.")
                return

            # Convert display coordinates to original image coordinates.
            x = int(x / ZOOM_FACTOR)
            y = int(y / ZOOM_FACTOR)

            self.center = (x, y)

            height, width = self.original_image.shape[:2]

            positive_windows = sample_positive_windows(x, y, width, height, WINDOW_SIZE)
            negative_window = sample_negative_window(positive_windows, width, height, WINDOW_SIZE)
            self.windows = positive_windows + [negative_window]

            for name, window in zip(list(C_LABELS), self.windows):
                print(f"{name}: {window}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.center is not None:
                self.center = None
                self.windows.clear()
                print("Current selection cancelled.")

    def save_and_next(self):
        if self.frame_index is None:
            return

        if not self.windows:
            print("Please select the droplet center first.")
            return

        if self.frame_index in self.existing_indices:
            print("This frame already exists in the label file.")
            return

        write_data(C_LABEL_PATH, self.frame_index, self.windows)

        self.existing_indices.add(self.frame_index)
        self.frame_index += 1

        self.load_frame()

    def go_back(self):
        if self.frame_index <= 0:
            print("Already at the first frame.")
            return

        # Remove the annotation of the previous frame.
        previous_index = self.frame_index - 1

        # Remove its saved annotation.
        self.delete_frame_records([previous_index, self.frame_index])
        print(f"Undo: frame {previous_index}")

        # Go back and reload.
        self.frame_index = previous_index
        self.load_frame()

    # -------------------- Display --------------------
    def draw_annotations(self):
        preview = self.display_image.copy()

        cv2.putText(
            preview, f"{self.frame_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2, cv2.LINE_AA)

        if self.center is not None:
            cx, cy = self.center
            cv2.drawMarker(preview, (cx * ZOOM_FACTOR, cy * ZOOM_FACTOR), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)

        for name, (x, y) in zip(list(C_LABELS), self.windows):
            x *= ZOOM_FACTOR
            y *= ZOOM_FACTOR
            size = WINDOW_SIZE * ZOOM_FACTOR
            cv2.rectangle(preview, (x, y), (x + size, y + size), (0, 255, 0), 1)
            cv2.putText(preview, name, (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return preview

    def delete_frame_records(self, frame_indexes: list[int]):
        """Delete the annotation record of specified frames."""
        if not C_LABEL_PATH.exists():
            return
        frame_indexes = set(frame_indexes)
        records = fileutils.CsvIO.read_from_csvfile(str(C_LABEL_PATH))
        fileutils.CsvIO.write_to_csvfile(
            str(C_LABEL_PATH), [x for x in records if int(x["index"]) not in frame_indexes])
        for frame_index in frame_indexes:
            self.existing_indices.discard(frame_index)
            print(f"Deleted annotation record for frame {frame_index}.")

    def _run(self):
        while True:
            if self.original_image is None:
                key = cv2.waitKey(0) & 0xFF
            else:
                cv2.imshow("image", self.draw_annotations())
                key = cv2.waitKey(20) & 0xFF

            if key == 27:  # Esc
                break
            elif key == 32:  # Space
                self.save_and_next()
            elif key == 8:  # BackSpace
                self.go_back()

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            print("Exited")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    annotator = ClassificationAnnotator()
    annotator.run()
