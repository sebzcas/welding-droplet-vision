import cv2
import random
import typing

from deeplearning.settings import *
from deeplearning.utils import fileutils

videoName2InfoMap = {videoInfo["name"]: videoInfo for videoInfo in VIDEOS_INFO}
videoName2SampleStrategyMap = {
    "20": {"low": 25, "high": 125, "mode": "a"},
    "21": {"low": 110, "high": 140, "mode": "a"},
    "24c": {"low": 20, "high": 50, "mode": "a"},
    "29": {"low": 20, "high": 40, "mode": "b"},
}


def random_event(p: float = 0) -> bool:
    return random.random() <= p


def sample_frame_index(low: int, high: int, start: int, end: int, mode: typing.Literal["a", "b", "c"] | str):
    if mode == "a":
        if random_event(.85):  # 按 0.85 的概率决定是否抽取周期内特定时段的编号
            return random.randrange(start + low, start + high)
        if random_event(.5):
            return random.randrange(start, start + low)
        return random.randrange(start + high, end)

    if mode == "b":
        if random_event(.85):
            if random_event(.5):
                return random.randrange(start, start + low)
            return random.randrange(start + high, end)
        return random.randrange(start + low, start + high)

    if mode == "c":
        return random.randrange(start, end)

    raise ValueError(f'No implementation for mode "{mode}"')


def sample_frame_route(video_names: list[str]) -> tuple[str, int]:  # 按概率生成图片的编号信息
    video_name = random.choice(video_names)  # 随机抽取原始图片集
    video_info = videoName2InfoMap[video_name]
    period_index = random.randrange(0, video_info["period_num"])  # 随机抽取周期
    frame_num_per_period = int(video_info["frame_num"] / video_info["period_num"])  # 计算每周期内所含帧数

    start, end = period_index * frame_num_per_period, (period_index + 1) * frame_num_per_period

    sample_strategy = videoName2SampleStrategyMap.get(video_name, {"low": 0, "high": 1, "mode": "c"})
    if not 0 <= sample_strategy["low"] < sample_strategy["high"] <= frame_num_per_period:
        raise ValueError("Invalid sampling interval")

    frame_index = sample_frame_index(
        low=sample_strategy["low"],
        high=sample_strategy["high"],
        start=start,
        end=end,
        mode=sample_strategy["mode"]
    )
    if frame_index > (frame_num := videoName2InfoMap[video_name]["frame_num"]) - 1:
        RuntimeError(
            f'[{video_name}] ({sample_strategy["mode"]}) frame_index:{frame_index} is out of 0 ~ {frame_num - 1}')

    return video_name, frame_index


def sample_frames(video_names: list[str], sample_num: int, with_replacement: bool = True):
    # frame_path = DATA_PATH / "frames"
    # sample_path = DATA_PATH / "sampled_frames"
    frame_path = FRAME_PATH
    sample_path = SAMPLE_PATH

    if not sample_path.exists():
        sample_path.mkdir(parents=True, exist_ok=True)
        print(f"Created path {sample_path}")
    else:
        raise ValueError(f"Sample path {sample_path} has already existed")

    sampling_history = []
    sampling_counter = dict()
    sampling_records = []
    for i in range(sample_num):
        selected_video_name, selected_frame_index = sample_frame_route(video_names)
        while not with_replacement and (selected_video_name, selected_frame_index) in sampling_history:
            selected_video_name, selected_frame_index = sample_frame_route(video_names)
        sampling_history.append((selected_video_name, selected_frame_index))

        source_filepath = frame_path / selected_video_name / f'{selected_frame_index}.png'
        image = cv2.imread(source_filepath)

        target_filepath = sample_path / f'{i}.png'
        cv2.imwrite(target_filepath, image)

        if selected_video_name not in sampling_counter:
            sampling_counter[selected_video_name] = {"num": 0, "frame2freq": dict()}
        sampling_counter[selected_video_name]["num"] += 1
        if selected_frame_index not in sampling_counter[selected_video_name]["frame2freq"]:
            sampling_counter[selected_video_name]["frame2freq"][selected_frame_index] = 0
        sampling_counter[selected_video_name]["frame2freq"][selected_frame_index] += 1

        sampling_records.append({
            "sample_index": i, "video_name": selected_video_name, "frame_index": selected_frame_index})

    fileutils.CsvIO.write_to_csvfile(str(SAMPLE_RECORDS_PATH), sampling_records)
    for video_name in sampling_counter:
        print(f'{video_name} contributed {sampling_counter[video_name]["num"]} frames')
        for frame_index in sampling_counter[video_name]["frame2freq"]:
            if (freq := sampling_counter[video_name]["frame2freq"][frame_index]) > 1:
                print(f" '- Frame {frame_index} appears more than once ({freq})")


if __name__ == '__main__':
    videoNames = ["20", "21", "22", "24c", "26c", "29", "33"]
    sample_frames(videoNames, 4000, with_replacement=False)
