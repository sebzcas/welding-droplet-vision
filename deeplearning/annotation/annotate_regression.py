import cv2
import random

from deeplearning.settings import *
from deeplearning.utils import fileutils


def sample_positive_windows(x: int, y: int, width: int, height: int, size: int = 32) -> list[tuple[int, int]]:
    windows = [
        (x - size // 2, y - size // 2),
        (random.randint(x - int(0.75 * size), x - int(0.25 * size)), random.randint(y - size, y - int(0.75 * size))),
        (random.randint(x - int(0.75 * size), x - int(0.25 * size)), random.randint(y - int(0.25 * size), y))
    ]
    windows = [(min(max(wx, 0), width - size), min(max(wy, 0), height - size)) for wx, wy in windows]
    return windows


def load_existing_indices(label_path) -> set[int]:
    """Read frame indices already saved in the label file."""
    if not label_path.exists():
        return set()
    with label_path.open("r", encoding="utf-8") as f:
        records = fileutils.CsvIO.to_records(f.read())
    return {int(x["index"]) for x in records}


def write_data(label_path, frame_index, bounds, windows):
    if len(bounds) != 5:
        raise ValueError("Exactly 5 boundary points are required.")

    if len(windows) != 3:
        raise ValueError("Exactly 3 crop windows are required.")

    bounds_text = ",".join(f"{x},{y}" for x, y in bounds)
    windows_text = ",".join(f"{x},{y}" for x, y in windows)

    record = f'{frame_index},{bounds_text},{windows_text}\n'

    content = ""
    if label_path.exists():
        with open(label_path, 'r', encoding="utf-8") as f:
            content = f.read()

    with open(label_path, 'a', encoding="utf-8") as f:
        if not content:
            f.write(",".join((
                "index",
                "upper_left_x", "upper_left_y",
                "upper_right_x", "upper_right_y",
                "bottom_x", "bottom_y",
                "left_x", "left_y",
                "right_x", "right_y",
                "full_x", "full_y",
                "upper_x", "upper_y",
                "lower_x", "lower_",
            )) + "\n")
        f.write(record)

    print(f"Saved: {record.strip()}")


class RegressionAnnotator:

    def __init__(self, start_index: int = None):
        self.existing_indices = load_existing_indices(R_LABEL_PATH)

        if start_index is None:
            self.frame_index = (list(self.existing_indices)[-1] + 1) if self.existing_indices else 0
        else:
            self.frame_index = start_index

        self.mode = "bounds"

        self.bounds = []
        self.windows = []

        self.original_image = None
        self.display_image = None

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.mouse_callback)

        self.load_frame()

    def load_frame(self):
        """Load and resize the current frame."""
        image_file = SAMPLE_PATH / f"{self.frame_index}.png"
        image = cv2.imread(str(image_file))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_file}")

        self.original_image = image
        self.display_image = cv2.resize(image, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)

        self.bounds.clear()
        self.windows.clear()
        self.mode = "bounds"

        print(f"\nCurrent frame: {self.frame_index}")
        print("Mode: boundary annotation")
        if self.frame_index in self.existing_indices:
            print("WARNING: This frame already exists in the label file.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "bounds":
                if len(self.bounds) >= 5:
                    print("5 boundary points already selected.")
                    return

                point = (int(x / ZOOM_FACTOR), int(y / ZOOM_FACTOR))
                self.bounds.append(point)
                print(f"{BOUNDARY_POINTS[len(self.bounds) - 1]}: {point}")

            elif self.mode == "windows":
                if self.windows:
                    print("Crop windows already generated. Right-click to reset.")
                    return

                center_x = int(x / ZOOM_FACTOR)
                center_y = int(y / ZOOM_FACTOR)
                height, width = self.original_image.shape[:2]
                self.windows = sample_positive_windows(center_x, center_y, width, height)

                for name, window in zip(R_WINDOWS, self.windows):
                    print(f"{name}: {window}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "bounds":
                if self.bounds:
                    removed = self.bounds.pop()
                    print("Undo boundary:", removed)

            elif self.mode == "windows":
                if self.windows:
                    self.windows.clear()
                    print("Undo crop windows.")

    def switch_mode(self):
        if self.mode == "bounds":
            if len(self.bounds) != 5:
                print("Please annotate all 5 boundary points first.")
                return

            self.mode = "windows"
            print("\nMode: crop window annotation")
            print("Click the droplet center.")
            return

        self.mode = "bounds"
        self.windows.clear()
        print("\nMode: boundary annotation")

    def save_and_next(self):
        if len(self.bounds) != 5:
            print("Cannot save: 5 boundary points are required.")
            return

        if len(self.windows) != len(R_WINDOWS):
            print(f"Cannot save: {len(R_WINDOWS)} crop windows are required.")
            return

        if self.frame_index in self.existing_indices:
            print("Cannot save: this frame already exists in the label file.")
            return

        write_data(R_LABEL_PATH, self.frame_index, self.bounds, self.windows)

        self.existing_indices.add(self.frame_index)
        self.frame_index += 1

        self.load_frame()

    def draw_annotations(self):
        """Draw annotations on a fresh copy, never modify source images."""
        preview = self.display_image.copy()

        # Draw the current frame index.
        cv2.putText(
            preview, f"{self.frame_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw boundary points.
        for i, (x, y) in enumerate(self.bounds):
            px = x * ZOOM_FACTOR
            py = y * ZOOM_FACTOR
            cv2.drawMarker(preview, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(preview, BOUNDARY_POINTS[i], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw crop windows.
        for name, (x, y) in zip(R_WINDOWS, self.windows):
            x *= ZOOM_FACTOR
            y *= ZOOM_FACTOR
            cv2.rectangle(preview, (x, y), (x + DISPLAY_SIZE, y + DISPLAY_SIZE), (0, 255, 0), 1)
            cv2.putText(preview, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return preview

    def delete_frame_records(self, frame_indexes: list[int]):
        """Delete the annotation record of specified frames."""
        if not R_LABEL_PATH.exists():
            return
        frame_indexes = set(frame_indexes)
        records = fileutils.CsvIO.read_from_csvfile(str(R_LABEL_PATH))
        fileutils.CsvIO.write_to_csvfile(
            str(R_LABEL_PATH), [x for x in records if int(x["index"]) not in frame_indexes])
        for frame_index in frame_indexes:
            self.existing_indices.discard(frame_index)
            print(f"Deleted annotation record for frame {frame_index}.")

    def go_back(self):
        if self.frame_index <= 0:
            print("Already at the first frame.")
            return

        # Remove the annotation of the previous frame.
        previous_index = self.frame_index - 1
        self.delete_frame_records([previous_index, self.frame_index])

        # Go back and reload.
        self.frame_index = previous_index
        self.load_frame()

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
            elif key == ord("5"):
                self.switch_mode()

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            print("Exited")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    annotator = RegressionAnnotator()
    annotator.run()
