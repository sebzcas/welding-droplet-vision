# Model Specification

## 1. Overview

The reconstructed deep-learning detection system uses two complementary CNN tasks:

1. Image-window classification to identify candidate droplet regions.
2. Boundary-coordinate regression to estimate droplet positions within candidate windows.

The reconstructed system contains one classification model and five independent regression models.

The CNN models provide experimental measurements of droplet position. These measurements can be interpreted alongside the theoretical dynamic model documented in [Original Research Pipeline](original_pipeline.md#5-theoretical-dynamic-modelling).

## 2. Input Representation

All CNN models operate on grayscale image windows with dimensions:

```text
1 × 32 × 32
```

The input image is represented as a single-channel tensor.

The models process small image windows rather than full-resolution video frames. Full-frame detection is performed by applying a sliding window to the resized image.

## 3. Classification CNN

### 3.1 Task

The classification model predicts one of four classes:

| Class   | Meaning              |
|---------|----------------------|
| `off`   | No droplet           |
| `full`  | Complete droplet     |
| `upper` | Upper droplet region |
| `lower` | Lower droplet region |

Its purpose is to identify candidate windows containing the droplet.

### 3.2 Architecture

The reconstructed classifier uses the following architecture:

| Layer    | Configuration                                | Output       |
|----------|----------------------------------------------|--------------|
| Input    | Grayscale image                              | 1 × 32 × 32  |
| Conv1    | 64 filters, 5 × 5, stride 1, padding 2, ReLU | 64 × 32 × 32 |
| MaxPool1 | 2 × 2                                        | 64 × 16 × 16 |
| LRN1     | Size 9, alpha 0.001, beta 0.75, k 1          | 64 × 16 × 16 |
| Conv2    | 64 filters, 5 × 5, stride 1, padding 2, ReLU | 64 × 16 × 16 |
| LRN2     | Size 9, alpha 0.001, beta 0.75, k 1          | 64 × 16 × 16 |
| MaxPool2 | 2 × 2                                        | 64 × 8 × 8   |
| Flatten  | 64 × 8 × 8                                   | 4,096        |
| FC1      | 384 units, ReLU                              | 384          |
| FC2      | 192 units, ReLU                              | 192          |
| FC3      | 4 output logits                              | 4            |

The class probabilities are obtained from the output logits using softmax during inference.

### 3.3 Loss and Optimization

The reconstructed classifier uses:

- Cross-entropy loss.
- L2 regularization on the weights of the first and second fully connected layers.
- Adam optimizer.
- Learning rate: 0.001.
- Batch size: 100.
- Random seed: 42.

The L2 regularization coefficient used in the reconstructed implementation is 0.002 for the regularized fully connected weights.

## 4. Regression CNNs

### 4.1 Task

The regression models predict the local `(x, y)` coordinates of droplet boundary points within a 32 × 32 image window.

Five independent models are trained:

| Model         | Predicted boundary   |
|---------------|----------------------|
| `upper_left`  | Upper-left boundary  |
| `upper_right` | Upper-right boundary |
| `bottom`      | Bottom boundary      |
| `left`        | Left boundary        |
| `right`       | Right boundary       |

Each model outputs two values:

```text
[x, y]
```

### 4.2 Architecture

The reconstructed regression models use the following architecture:

| Layer    | Configuration                                | Output       |
|----------|----------------------------------------------|--------------|
| Input    | Grayscale image                              | 1 × 32 × 32  |
| Conv1    | 64 filters, 5 × 5, stride 1, padding 2, ReLU | 64 × 32 × 32 |
| MaxPool1 | 2 × 2                                        | 64 × 16 × 16 |
| Conv2    | 64 filters, 5 × 5, stride 1, padding 2, ReLU | 64 × 16 × 16 |
| MaxPool2 | 2 × 2                                        | 64 × 8 × 8   |
| Flatten  | 64 × 8 × 8                                   | 4,096        |
| FC1      | 384 units, ReLU                              | 384          |
| FC2      | 192 units, ReLU                              | 192          |
| FC3      | 2 coordinate outputs                         | 2            |

Unlike the reconstructed classifier, the regression models do not use LRN layers.

### 4.3 Loss and Optimization

Each regression model uses:

- Mean squared error (MSE) loss.
- Adam optimizer.
- Learning rate: 0.001.
- Batch size: 80.
- Random seed: 42.

Each target is trained independently using its corresponding dataset.

## 5. Training Configuration

| Setting                     | Classification                    | Regression  |
|-----------------------------|-----------------------------------|-------------|
| Input size                  | 1 × 32 × 32                       | 1 × 32 × 32 |
| Optimizer                   | Adam                              | Adam        |
| Learning rate               | 0.001                             | 0.001       |
| Batch size                  | 100                               | 80          |
| Epochs in reconstructed run | 20                                | 20          |
| Random seed                 | 42                                | 42          |
| Loss                        | Cross-entropy + L2 regularization | MSE         |

The reconstructed training implementation uses the configured training and validation subsets and evaluates the final models on their respective test subsets.

The training run's console output is preserved in `data/results/deep-learning/checkpoints/training.log`.

## 6. Original Research Results

The original research reported:

| Task                            | Original reported result |
|---------------------------------|-------------------------:|
| Classification accuracy         |        Approximately 96% |
| Upper-left coordinate accuracy  |        Approximately 96% |
| Upper-right coordinate accuracy |        Approximately 96% |
| Bottom coordinate accuracy      |        Approximately 93% |
| Left coordinate accuracy        |        Approximately 93% |
| Right coordinate accuracy       |        Approximately 93% |

For coordinate regression, the original research defined accuracy as a prediction within two pixels of the ground-truth coordinate.

These figures describe the historical experiments. They are not the metrics of the reconstructed PyTorch training run.

## 7. Reconstructed Training Results

### 7.1 Classification

The reconstructed classifier was trained using 12,000 samples and evaluated on 4,000 test samples.

| Metric                    | Final result |
|---------------------------|-------------:|
| Training loss at epoch 20 |     0.094193 |
| Test cross-entropy loss   |     0.049914 |
| Test accuracy             |       98.50% |

The training loss includes L2 regularization, whereas the reported final test loss is cross-entropy loss only.

### 7.2 Regression

The following table reports the final test metrics from the reconstructed PyTorch training run.

| Target        | Train samples | Test samples | Validation samples | Test MSE | Test MAE (px) | Within 2 px |
|---------------|--------------:|-------------:|-------------------:|---------:|--------------:|------------:|
| `upper_left`  |         9,200 |          400 |                 55 | 2.230845 |        1.0599 |      94.00% |
| `upper_right` |         9,000 |          100 |                 80 | 1.487209 |        0.8605 |      98.00% |
| `bottom`      |         7,680 |          240 |                 71 | 2.183379 |        1.0113 |      93.75% |
| `left`        |        10,560 |          400 |                 79 | 1.826351 |        0.8533 |      95.25% |
| `right`       |        10,880 |           80 |                 77 | 1.141865 |        0.7962 |      96.25% |

MSE and MAE are reported per coordinate.

The within-two-pixel metric is a sample-level accuracy measure: both predicted coordinates must satisfy the two-pixel tolerance. It follows the reconstructed implementation's integer-coordinate conversion and tolerance calculation.

The regression test-set sizes vary substantially between targets. Metrics should therefore be interpreted together with their sample counts rather than compared as if all targets had equally sized test sets.

## 8. Detection Pipeline

### 8.1 Sliding-window Classification

The detector scans each resized grayscale frame using a 32 × 32 window.

The current implementation uses:

- Window size: 32 × 32 pixels.
- Sliding stride: 5 pixels.
- Scan margin: 15 pixels.

The classifier identifies candidate windows classified as `full`.

### 8.2 Coordinate Regression

Candidate windows classified as `full` are passed to the corresponding regression models.

Each model predicts a local coordinate. The window's top-left position is added to the prediction to obtain the coordinate in the resized full-frame image.

Predictions outside the image boundaries are discarded.

### 8.3 Prediction Consolidation

Multiple windows may generate predictions for the same boundary. The reconstructed detector consolidates these predictions using target-specific spatial clustering and selection rules.

| Target        | Consolidation rule                                                                                                                                     |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `upper_left`  | Cluster nearby points using squared distance threshold 2; select the point with maximum y-coordinate from the largest cluster                          |
| `upper_right` | Cluster nearby points using squared distance threshold 2; select the point with minimum y-coordinate from the largest cluster                          |
| `bottom`      | Select the point with maximum y-coordinate from all valid predictions                                                                                  |
| `left`        | Cluster nearby points using squared distance threshold 2; select the point with minimum x-coordinate and maximum y-coordinate from the largest cluster |
| `right`       | Cluster nearby points using squared distance threshold 8; select the point with maximum x-coordinate and maximum y-coordinate from the largest cluster |

The final coordinates are rounded to integer pixel positions.

### 8.4 Output

For each video, the detector generates a CSV file containing:

- Frame index.
- Upper-left x and y.
- Upper-right x and y.
- Bottom x and y.
- Left x and y.
- Right x and y.

Frames without a detected droplet may contain empty coordinate fields.

The outputs are used to generate annotated GIFs and boundary displacement plots.

## 9. Relationship to the Theoretical Dynamic Model

The CNN models estimate droplet boundary positions from experimental images. The theoretical dynamic model describes the expected droplet displacement over one transfer cycle.

The original research used the CNN-derived bottom-boundary trajectory for its principal theoretical comparison.

The theoretical model is a physical reference rather than a component of the CNN inference pipeline. It does not participate in classification, regression, or prediction consolidation.

For a direct comparison, the experimental displacement curve and theoretical model must use consistent:

- Time origins.
- Displacement reference positions.
- Physical units.
- Transfer-cycle intervals.

The original research reported approximately 90% agreement for one representative theoretical-to-experimental comparison. This historical result is separate from the reconstructed CNN training metrics.

See [Original Research Pipeline](original_pipeline.md#5-theoretical-dynamic-modelling) for the model derivation and [Theoretical Model and Experimental Comparison](original_pipeline.md#6-theoretical-model-and-experimental-comparison) for the historical comparison.

## 10. Checkpoints and Training Records

The reconstructed model checkpoints and training record are stored under:

[`data/results/deep-learning/checkpoints/`](../data/results/deep-learning/checkpoints/)

The checkpoint package contains:

- One classification model.
- Five regression models.
- `training.log`, containing captured console output from the training runs.

The detection scripts support explicit checkpoint-path arguments. If paths are not provided, the scripts use their configured default checkpoint locations under `deeplearning/training/runs/`.

See the [Deep-learning Module README](../deeplearning/README.md) for the exact command-line arguments and execution instructions.

## 11. Evaluation Limitations

Several factors should be considered when interpreting the reconstructed metrics:

1. The regression datasets use sample-based splitting. Samples from the same source frame may appear in different subsets.
2. The five regression targets have different test-set sizes.
3. The within-two-pixel metric is a sample-level coordinate-tolerance measure, not a conventional regression accuracy metric.
4. The classifier's training loss includes L2 regularization, while its reported final test loss does not.
5. The reconstructed PyTorch implementation is not an exact reproduction of the original TensorFlow execution environment.
6. The original theoretical comparison used a representative CNN-derived bottom-boundary trajectory. It does not establish equivalent agreement for every reconstructed video or boundary target.

These limitations are documented to make the experimental results interpretable and to distinguish the reconstructed implementation from the historical research results.
