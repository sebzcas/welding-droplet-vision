# Dataset Specification

## 1. Overview

The dataset consists of high-speed welding videos recorded during piezoelectric actuator-assisted Gas Metal Arc Welding (GMAW) experiments.

The videos contain different droplet-transfer conditions and are used to construct image-window datasets for CNN classification and boundary-coordinate regression.

The repository distinguishes between:

- Original experimental video recordings.
- Extracted video frames.
- Randomly sampled frames.
- Manually annotated image windows and boundary coordinates.
- Constructed classification and regression datasets.
- Detection outputs and visualization results.

Original video recordings are distributed separately through GitHub Releases where redistribution is permitted. GIF previews are available in `data/video-samples/`.

## 2. Original Video Recordings

The original research describes seven video sequences containing a total of 14,074 frames.

The reconstructed repository uses the following video IDs and original filenames:

| Video ID  | Original filename  |     Frames | Transfer periods |
|-----------|--------------------|-----------:|-----------------:|
| 20        | `20_.avi`          |      3,000 |               20 |
| 21        | `21_.avi`          |      3,000 |               20 |
| 22        | `22_.avi`          |      3,001 |                1 |
| 24c       | `24__clip.wmv`     |      1,036 |               14 |
| 26c       | `26__clip.wmv`     |        737 |                3 |
| 29        | `29_.avi`          |      3,000 |               60 |
| 33        | `Y4 Camera33_.avi` |        300 |                9 |
| **Total** |                    | **14,074** |                  |

The transfer-period counts were obtained by manually examining the videos. The shorter video IDs are used consistently in the reconstructed code and output directories.

The original filenames are retained to identify the source recordings.

## 3. Frame Extraction and Preprocessing

The original research describes source frames with a resolution of 500 × 352 pixels.

The preprocessing workflow reduces the image dimensions to 40% of the original resolution, resulting in 200 × 140 images.

The reconstructed implementation uses OpenCV to extract frames and resize them using the interpolation method configured in the preprocessing code.

The project uses separate directories for:

- Original videos.
- Extracted frames.
- Sampled frames.
- Constructed datasets.
- Detection outputs.

Refer to the [Deep-learning Module README](../deeplearning/README.md) for the current directory configuration and execution instructions.

## 4. Sample Selection

### 4.1 Original Research Strategy

The original research selected 4,000 frames from the complete set of 14,074 frames.

Its sampling strategy aimed to preserve variation across experimental conditions and droplet-transfer cycles:

1. Select among the seven video sequences with equal probability.
2. Select a transfer cycle within the chosen video.
3. Sample more frequently during the droplet-detachment and falling stage than during the droplet-dwelling stage.

The intention was to avoid a dataset dominated by frames in which the droplet remained attached to the wire.

### 4.2 Reconstructed Sampling Implementation

The current implementation also constructs a 4,000-frame sampled set without replacement.

Sampling is performed across video sequences, transfer cycles, and frame indices. The implementation uses per-video sampling settings to constrain or adjust the frame-selection range and sampling mode.

The selected samples are recorded in `sampling_records.csv`, which associates each sample index with its source video and frame index.

The current implementation should be understood as the reconstructed sampling procedure implemented in the repository. It should not be assumed to reproduce every probability and sampling decision of the original research exactly.

## 5. Image-window Annotation

The sampled frames are used to construct two types of manually annotated datasets.

### 5.1 Classification Annotation

Each sampled frame contributes four 32 × 32 image windows:

| Label   | Description                                |
|---------|--------------------------------------------|
| `off`   | Window without the droplet                 |
| `full`  | Window containing the complete droplet     |
| `upper` | Window containing the upper droplet region |
| `lower` | Window containing the lower droplet region |

Each class contains 4,000 samples, resulting in 16,000 classification samples.

The annotation coordinates identify the locations of the corresponding windows in the source frame.

### 5.2 Regression Annotation

The regression annotation records five droplet boundary points:

- `upper_left`
- `upper_right`
- `bottom`
- `left`
- `right`

The annotation also records the positions of the `full`, `upper`, and `lower` windows.

A regression sample is retained for a particular target when the corresponding image window contains that target point. Consequently, the five target-specific datasets have different sample counts.

Coordinates are represented as `(x, y)` pairs in the reconstructed dataset.

## 6. Dataset Construction

### 6.1 Classification Dataset

The reconstructed classification dataset contains:

| Class     |      Total |   Training |      Test |
|-----------|-----------:|-----------:|----------:|
| `off`     |      4,000 |      3,000 |     1,000 |
| `full`    |      4,000 |      3,000 |     1,000 |
| `upper`   |      4,000 |      3,000 |     1,000 |
| `lower`   |      4,000 |      3,000 |     1,000 |
| **Total** | **16,000** | **12,000** | **4,000** |

The reconstructed implementation groups classification samples by source frame during splitting, so the four windows originating from the same frame are assigned to the same subset.

### 6.2 Original Regression Dataset Statistics

The original research constructed five target-specific regression datasets from 12,000 selected image windows.

Since not every window contained every boundary point, the resulting datasets had different sizes:

| Target      | Total samples | Training | Test |
|-------------|--------------:|---------:|-----:|
| Upper-left  |         9,600 |    9,200 |  400 |
| Upper-right |         9,100 |    9,000 |  100 |
| Bottom      |         7,920 |    7,680 |  240 |
| Left        |        10,960 |   10,560 |  400 |
| Right       |        10,960 |   10,880 |   80 |

These figures describe the historical dataset statistics and should not be confused with the current reconstructed training-run statistics.

### 6.3 Reconstructed Regression Datasets

The reconstructed regression datasets are separated by boundary target.

| Target        | Training | Test | Validation |
|---------------|---------:|-----:|-----------:|
| `upper_left`  |    9,200 |  400 |         55 |
| `upper_right` |    9,000 |  100 |         80 |
| `bottom`      |    7,680 |  240 |         71 |
| `left`        |   10,560 |  400 |         79 |
| `right`       |   10,880 |   80 |         77 |

The target-specific sample counts differ because not every candidate image window contains every boundary point.

### 6.4 Split Strategy and Limitation

The reconstructed classification dataset uses source-frame-based grouping.

The regression datasets use sample-based splitting (`R_SPLIT_MODE = "sample"`), consistent with the original implementation's sample-oriented experimental procedure.

Under sample-based splitting, individual samples are assigned to training, testing, and validation subsets independently of their source video frames. Therefore, different image windows originating from the same source frame may appear in different subsets.

This may introduce correlations between the subsets and should be considered when interpreting the reconstructed regression metrics. A frame-based split could provide a stricter evaluation of generalization to unseen frames, but the reported reconstructed results retain the sample-based procedure.

## 7. Metadata and File Formats

The repository stores metadata and annotations in CSV format.

| File                        | Description                                                             |
|-----------------------------|-------------------------------------------------------------------------|
| `videos_info.csv`           | Video IDs, original filenames, frame counts, and transfer-period counts |
| `sampling_records.csv`      | Mapping between sampled-frame indices and source video/frame indices    |
| `classification_labels.csv` | Classification-window coordinates and labels                            |
| `regression_labels.csv`     | Boundary-point coordinates and regression-window coordinates            |

The constructed datasets are stored in NumPy `.npz` files:

- `classification_dataset.npz`
- `regression_datasets.npz`

The regression targets are stored as `(x, y)` coordinates in the reconstructed implementation.

## 8. Detection Outputs

The repository contains detection outputs for the following video sequences.

**Deep learning:** `20`, `21`, `22`, `24c`, `26c`, `29`, `33`.

**Traditional vision:** `20`, `24c`, `26c`, `33`.

Each available result directory contains a detection CSV, an annotated GIF, and boundary displacement plots.

The deep-learning detection CSV records the frame index and the predicted coordinates of the five boundary targets. Frames without a detected droplet may contain empty coordinate fields.

Traditional-vision CSV files contain the three boundary outputs available in that implementation: upper-left, upper-right, and bottom.

## 9. Theoretical Model and Experimental Data

The original research used the experimentally detected droplet-bottom displacement curve to compare against the four-stage theoretical dynamic model.

The theoretical model describes the displacement over one complete droplet-transfer cycle. Its initial time and displacement can be adjusted to align it with a selected experimental cycle.

The historical theoretical-to-experimental comparison is documented in [Original Research Pipeline](original_pipeline.md#6-theoretical-model-and-experimental-comparison) and illustrated in [`docs/figures/fig-4-9-theoretical-experimental-comparison.png`](figures/fig-4-9-theoretical-experimental-comparison.png).

The original approximately 90% agreement refers to the representative historical comparison. It should not be interpreted as a newly measured metric for every reconstructed video or for all boundary targets.

The repository's per-video displacement plots are available under:

- `data/results/deep-learning/{video_id}/plots/`
- `data/results/traditional-vision/{video_id}/plots/`

These plots preserve the reconstructed detection outputs. A direct comparison with the theoretical model requires consistent time alignment, displacement reference, and physical units.

## 10. Data Availability

The original experimental videos are distributed separately through GitHub Releases, subject to permission to redistribute the recordings.

GIF previews are included in `data/video-samples/`. They are provided for visual reference and are not substitutes for the original video recordings.

The repository does not require all extracted frame PNG files to be committed. Original videos and constructed datasets can be distributed as separate release assets when appropriate.

## 11. Reproducibility Notes

To reconstruct the datasets:

1. Obtain the original video recordings.
2. Extract and resize the video frames.
3. Sample 4,000 frames using the configured sampling procedure.
4. Annotate classification windows and regression coordinates.
5. Build the classification and five regression datasets.
6. Apply the configured dataset split strategies.

See the [Deep-learning Module README](../deeplearning/README.md) for environment setup, commands, and implementation details.
