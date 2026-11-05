import cv2
import pandas as pd

from deeplearning.settings import *


class BaseViewer:

    def __init__(self, csv_path, window_name, start_index=0):
        self.df = pd.read_csv(csv_path)

        if "index" not in self.df.columns:
            raise ValueError("CSV must contain an 'index' column.")

        # Use frame index as the actual key, not CSV row number.
        self.df["index"] = self.df["index"].astype(int)
        self.df = self.df.set_index("index", drop=False)

        self.frame_indices = sorted(self.df.index.unique().tolist())

        if not self.frame_indices:
            raise ValueError("CSV contains no samples.")

        self.window_name = window_name
        self.current_position = self._find_position(start_index)

        self.frame_index = None
        self.display_image = None
        self.windows = []

    def _find_position(self, frame_index):
        """Find the requested frame, or the nearest following frame."""
        for i, index in enumerate(self.frame_indices):
            if index >= frame_index:
                return i

        return len(self.frame_indices) - 1

    def _load_frame(self, frame_index):
        image_path = SAMPLE_PATH / f"{frame_index}.png"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read source frame: {image_path}")

        self.display_image = cv2.resize(image, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)

        self.frame_index = frame_index

    def _show_current(self):
        frame_index = self.frame_indices[self.current_position]
        row = self.df.loc[frame_index]

        self._load_frame(frame_index)
        self.load_annotations(row)

        cv2.imshow("image", self.draw_annotations())

        print(
            f"Frame: {frame_index} "
            f"({self.current_position + 1}/{len(self.frame_indices)})"
        )

    def load_annotations(self, row):
        raise NotImplementedError

    def draw_annotations(self):
        raise NotImplementedError

    def run(self):
        """Navigate through frames using keyboard."""
        self._show_current()

        while True:
            key = cv2.waitKeyEx(0)

            # ESC: quit
            if key == 27:
                break

            # Space: next frame
            elif key == 32:
                if self.current_position < len(self.frame_indices) - 1:
                    self.current_position += 1
                    self._show_current()

            # Backspace: previous frame
            elif key in (8, 127, 65288):
                if self.current_position > 0:
                    self.current_position -= 1
                    self._show_current()

        cv2.destroyWindow(self.window_name)


class ClassificationViewer(BaseViewer):

    def __init__(self, csv_path, start_index=0):
        super().__init__(
            csv_path,
            window_name="Classification Dataset Viewer",
            start_index=start_index
        )

        self.center = None
        self.windows = []

    def load_annotations(self, row):

        self.center = None

        self.windows = [
            (int(row["full_x"]), int(row["full_y"])),
            (int(row["upper_x"]), int(row["upper_y"])),
            (int(row["lower_x"]), int(row["lower_y"])),
            (int(row["off_x"]), int(row["off_y"])),
        ]

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


class RegressionViewer(BaseViewer):

    def __init__(self, csv_path, start_index=0):
        super().__init__(
            csv_path,
            window_name="Regression Dataset Viewer",
            start_index=start_index
        )

        self.bounds = []
        self.windows = []

    def load_annotations(self, row):

        self.bounds = [
            (int(row["upper_left_x"]), int(row["upper_left_y"])),
            (int(row["upper_right_x"]), int(row["upper_right_y"])),
            (int(row["bottom_x"]), int(row["bottom_y"])),
            (int(row["left_x"]), int(row["left_y"])),
            (int(row["right_x"]), int(row["right_y"]))
        ]

        self.windows = [
            (int(row["full_x"]), int(row["full_y"])),
            (int(row["upper_x"]), int(row["upper_y"])),
            (int(row["lower_x"]), int(row["lower_y"]))
        ]

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
            color = TARGET_COLORS[i]
            cv2.drawMarker(preview, (px, py), color, cv2.MARKER_CROSS, 12, 2)
            cv2.putText(preview, BOUNDARY_POINTS[i], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw crop windows.
        for name, (x, y) in zip(R_WINDOWS, self.windows):
            x *= ZOOM_FACTOR
            y *= ZOOM_FACTOR
            cv2.rectangle(preview, (x, y), (x + DISPLAY_SIZE, y + DISPLAY_SIZE), (0, 255, 0), 1)
            cv2.putText(preview, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return preview


if __name__ == "__main__":
    # viewer = ClassificationViewer(csv_path=C_LABEL_PATH, start_index=0)
    viewer = RegressionViewer(csv_path=R_LABEL_PATH, start_index=0)
    viewer.run()
