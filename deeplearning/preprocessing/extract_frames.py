import cv2
import tqdm

from deeplearning.settings import *

videoName2InfoMap = {videoInfo["name"]: videoInfo for videoInfo in VIDEOS_INFO}


def zoom_image(img: cv2.typing.MatLike, k: float) -> cv2.typing.MatLike:
    height, width = img.shape[:2]
    if k > 0:
        img = cv2.resize(img, (int(k * width), int(k * height)), interpolation=cv2.INTER_CUBIC)
    return img


def extract_frames(video_name: str):
    # video_path = DATA_PATH / "videos" / videoName2InfoMap[video_name]["filename"]
    # frame_path = DATA_PATH / "frames" / video_name
    video_path = VIDEO_PATH
    frame_path = FRAME_PATH

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not frame_path.exists():
        frame_path.mkdir(parents=True, exist_ok=True)
        print(f"Created path {frame_path}")
    else:
        raise ValueError(f"Frame path {frame_path} has already existed")

    try:
        k = 0
        with tqdm.tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
            while True:
                success, image = cap.read()
                if not success:
                    break

                # Display video
                cv2.imshow(video_name, image)

                # Resize image
                image = zoom_image(image, 0.4)

                # Save image
                target_filepath = frame_path / f"{k}.png"
                success = cv2.imwrite(target_filepath, image)
                if not success:
                    raise RuntimeError(f"Failed to save image: {target_filepath}")

                k += 1
                pbar.update(1)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    videoNames = ["20", "21", "22", "24c", "26c", "29", "33"]
    for videoName in videoNames:
        extract_frames(videoName)
