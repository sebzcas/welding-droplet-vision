import cv2
import re
from PIL import Image
from tqdm import tqdm
from pathlib import Path


def frames_to_video(frames_dir: Path, output_path: Path, fps: int = 15, scale: float = 0.5):
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find PNG files and sort by numeric filename.
    frame_paths = list(frames_dir.glob("*.png"))
    frame_paths.sort(key=lambda p: int(re.search(r"\d+", p.stem).group()))

    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    # Read the first frame to determine output dimensions.
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Cannot read image: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    output_size = (int(width * scale), int(height * scale))

    # Initialize video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)

    if not writer.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter.")

    try:
        for frame_path in tqdm(frame_paths, desc="Creating video", unit="frame"):
            frame = cv2.imread(str(frame_path))

            if frame is None:
                print(f"Warning: Cannot read {frame_path}")
                continue

            frame = cv2.resize(frame, output_size)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Video saved to: {output_path}")


def video_to_gif(
    video_path: str | Path,
    output_path: str | Path,
    start_sec: float = 0,
    end_sec: float | None = 10,
    target_fps: int = 20,
    speed: float = 1.5,
    max_width: int = 800,
    grayscale: bool = True,
    roi: tuple[int, int, int, int] | None = None,
    loop: int = 0,
) -> None:
    """
    Convert a video segment to GIF.

    Args:
        video_path: Input video path.
        output_path: Output GIF path.
        start_sec: Starting timestamp in seconds.
        end_sec: Ending timestamp in seconds. None means video end.
        target_fps: Output GIF frame rate.
        speed: Playback speed multiplier.
        max_width: Maximum output width. None means no resizing.
        grayscale: Whether to convert frames to grayscale.
        roi: Optional crop region (x1, y1, x2, y2).
        loop: Number of loops. 0 means infinite looping.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    if target_fps <= 0 or speed <= 0:
        raise ValueError("target_fps and speed must be positive.")

    if start_sec < 0:
        raise ValueError("start_sec must be non-negative.")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        cap.release()
        raise RuntimeError("Invalid source video FPS.")

    video_duration = total_frames / source_fps

    if end_sec is None:
        end_sec = video_duration

    if end_sec <= start_sec:
        cap.release()
        raise ValueError("end_sec must be greater than start_sec.")

    end_sec = min(end_sec, video_duration)

    print(f"Video: {video_path.name}")
    print(f"Source FPS: {source_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {video_duration:.2f}s")
    print(f"Segment: {start_sec:.2f}s - {end_sec:.2f}s")

    # Calculate sampling interval
    frame_interval = max(1, round(source_fps * speed / target_fps))

    start_frame = round(start_sec * source_fps)
    end_frame = round(end_sec * source_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []

    try:
        for frame_index in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_index - start_frame) % frame_interval != 0:
                continue

            # Crop ROI
            if roi is not None:
                x1, y1, x2, y2 = roi
                frame = frame[y1:y2, x1:x2]

            # Convert color space
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize
            height, width = frame.shape[:2]

            if max_width is not None and width > max_width:
                new_height = round(height * max_width / width)
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)

            frames.append(frame)

    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No valid frames extracted.")

    # Convert to PIL images
    if grayscale:
        palette = Image.new("P", (1, 1))
        palette.putpalette([i for i in range(256) for _ in range(3)])

        gif_frames = [
            Image.fromarray(frame).convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
            for frame in frames
        ]
    else:
        gif_frames = [
            Image.fromarray(frame).quantize(colors=256, dither=Image.Dither.NONE)
            for frame in frames
        ]

    # Save GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / target_fps),
        loop=loop,
        optimize=True
    )

    file_size = output_path.stat().st_size / 1024 / 1024

    print(f"\nGIF saved: {output_path}")
    print(f"Frames: {len(gif_frames)}")
    print(f"Resolution: {gif_frames[0].size}")
    print(f"Output FPS: {target_fps}")
    print(f"Playback speed: {speed}x")
    print(f"File size: {file_size:.2f} MB")


if __name__ == "__main__":
    videoNames = ["20", "21", "22", "24c", "26c", "29", "33"]
    # from deeplearning.settings import *
    # for videoName in videoNames:
    #     frames_to_video((
    #         frames_dir=OUTPUTS_PATH / videoName,
    #         output_path=OUTPUTS_PATH / f"{videoName}.mp4",
    #         fps=15,
    #         scale=.7,
    #     )
    # video_to_gif(
    #     video_path=DATA_PATH / "outputs-traditional_vision" / "33.mp4",
    #     output_path=DATA_PATH / "outputs-traditional_vision" / "33.gif",
    #     start_sec=5, end_sec=10,
    #     speed=1,
    #     grayscale=False
    # )
