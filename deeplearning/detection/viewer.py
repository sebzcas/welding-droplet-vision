import pandas as pd
import matplotlib.pyplot as plt
import typing

from deeplearning.settings import *


def show_detection_results(
        video_name: str,
        target: typing.Literal["upper_left", "upper_right", "bottom", "left", "right"],
        start_index: int = 0,
        end_index: int = None,
        save_to: str = None,
        show: bool = True
):
    df = pd.read_csv(OUTPUTS_PATH / f"{video_name}.csv")

    if end_index is None:
        end_index = df["frame_index"].max()

    df = df.loc[(df["frame_index"] >= start_index) & (df["frame_index"] <= end_index)].copy()
    print(df[["frame_index", f"{target}_x", f"{target}_y"]])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["frame_index"], df[f"{target}_x"], color="darkred", linewidth=1)
    axes[0].set_ylabel("X coordinate (pixel)")
    axes[0].set_title(f"[{video_name}] {target} - X")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["frame_index"], df[f"{target}_y"], color="darkblue", linewidth=1)
    axes[1].set_ylabel("Y coordinate (pixel)")
    axes[1].set_xlabel("Frame index")
    axes[1].set_title(f"[{video_name}] {target} - Y")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")


if __name__ == '__main__':
    videoNames = ["20", "21", "22", "24c", "26c", "29", "33"]
    show_detection_results("20", "bottom")
    # for videoName in videoNames:
    #     for targetName in BOUNDARY_POINTS:
    #         saveTo = OUTPUTS_PATH / "plots" / f"{videoName}-{targetName}.png"
    #         saveTo.parent.mkdir(parents=True, exist_ok=True)
    #         show_detection_results(videoName, targetName, save_to=str(saveTo), show=False)
