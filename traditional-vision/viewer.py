import pandas as pd
import matplotlib.pyplot as plt
import typing
from pathlib import Path


def show_detection_results(
        path: Path,
        video_name: str,
        target: typing.Literal["upper_left", "upper_right", "bottom"],
        start_index: int = 0,
        end_index: int = None,
        save_to: str = None,
        show: bool = True
):
    df = pd.read_csv(path / f"{video_name}.csv")

    if end_index is None:
        end_index = len(df) - 1

    df = df.loc[(df.index >= start_index) & (df.index <= end_index)].copy()
    if target == "bottom":
        df[f"{target}_x"] = None
    print(df)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df.index, df[f"{target}_x"], color="darkred", linewidth=1)
    axes[0].set_ylabel("X coordinate (pixel)")
    axes[0].set_title(f"[{video_name}] {target} - X")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df[f"{target}_y"], color="darkblue", linewidth=1)
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
    DATA_PATH = Path(r"D:\Projects\welding-droplet-vision-data")
    videoNames = ["20", "24c", "26c", "33"]
    # for videoName in videoNames:
    #     for targetName in ("upper_left", "upper_right", "bottom"):
    #         saveTo = DATA_PATH / "outputs-traditional_vision" / "plots" / f"{videoName}-{targetName}.png"
    #         saveTo.parent.mkdir(parents=True, exist_ok=True)
    #         show_detection_results(
    #             DATA_PATH / "outputs-traditional_vision", videoName, targetName, save_to=str(saveTo), show=False)
