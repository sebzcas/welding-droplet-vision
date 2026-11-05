# Research Summary

## 1. Research Background

Gas Metal Arc Welding (GMAW) is widely used in industrial welding because of its relatively high efficiency and broad applicability. However, conventional GMAW droplet transfer is affected by welding parameters such as voltage and current. The resulting transfer process can be difficult to control precisely, particularly in applications requiring stable and accurately controlled welding.

Piezoelectric actuator-assisted GMAW introduces an active wire-feeding control mechanism. A piezoelectric actuator periodically clamps and releases the welding wire. During clamping, the wire deforms and stores elastic energy. When released, the wire rebounds and generates transient motion that helps detach the droplet from the wire tip.

Understanding this process requires both a theoretical model of the transfer dynamics and accurate experimental measurements of droplet displacement.

High-speed welding images provide a way to observe the transfer process. However, intense arc-light interference, blurred boundaries, and fragmented contours make reliable image-based measurements difficult.

## 2. Research Objectives

The research pursued three connected objectives:

1. Develop computer-vision algorithms to detect droplet boundaries in experimental GMAW images and obtain displacement curves.
2. Establish a theoretical dynamic model describing the droplet-transfer process under piezoelectric actuator control.
3. Compare the theoretical displacement model with experimentally measured displacement curves to examine the model's validity.

Both traditional image processing and CNN-based detection were investigated. The CNN-based method was used to obtain the experimental droplet-position measurements for the principal theoretical-model comparison.

## 3. Methodology

### 3.1 Traditional Image Processing

A C++ and OpenCV-based image-processing pipeline was developed as a supplementary detection method.

The pipeline included:

- Grayscale conversion.
- Gaussian filtering.
- Canny edge detection.
- Contour tracing and reconstruction.
- Contour extension to address missing boundary segments.
- Extraction of the droplet's upper-left, upper-right, and bottom boundary positions.

The original research reported that the method required substantial manual parameter selection and was sensitive to arc-light interference. Its resulting displacement curves contained considerable noise and could fail to preserve important motion characteristics.

### 3.2 CNN-based Droplet Detection

The deep-learning approach used two stages.

**Classification network**

A CNN classified 32 × 32 image windows into four categories:

- `off`: no droplet.
- `full`: complete droplet.
- `upper`: upper droplet region.
- `lower`: lower droplet region.

The classifier was used to identify candidate droplet regions in larger images.

**Regression networks**

Five target-specific CNN regression models predicted two-dimensional boundary coordinates:

- Upper-left boundary.
- Upper-right boundary.
- Bottom boundary.
- Left boundary.
- Right boundary.

The predicted local coordinates were mapped back to the full-frame coordinate system. Multiple candidate predictions were consolidated to obtain the final boundary estimates.

The reconstructed implementation retains this classification-and-regression design using PyTorch.

### 3.3 Theoretical Dynamic Modelling

The theoretical model describes a complete droplet-transfer cycle through four stages:

1. **Droplet dwelling:** The wire feeding speed and melting speed are balanced, so the droplet remains approximately stationary.
2. **Wire energy storage:** The piezoelectric actuator clamps the wire. The wire tip stops advancing while the wire continues feeding, causing elastic deformation and energy accumulation.
3. **Wire rebound:** The actuator releases the wire. The stored elastic energy is released, and the wire rebounds toward its original shape.
4. **Droplet detachment:** The wire tip rapidly decelerates. The droplet continues moving due to inertia, overcomes the surface tension of the liquid connection, and detaches.

The analysis combines wire deformation, an equivalent spring-mass representation of wire rebound, and a simplified oscillatory model of droplet detachment.

The four stage-specific displacement expressions are integrated into a piecewise theoretical displacement function.

### 3.4 Theoretical-to-Experimental Comparison

The theoretical model is compared with an experimentally measured droplet-bottom displacement trajectory.

The original research selected a representative transfer cycle from the CNN-derived experimental displacement curve. The initial time and displacement were adjusted to align the theoretical and experimental trajectories.

The comparison was presented in the original research as a theoretical-versus-experimental displacement plot. The reported overall agreement was approximately 90%.

This comparison is an important part of the research: the computer-vision algorithms provide experimental measurements, while the theoretical model provides a physical reference for interpreting those measurements.

The historical comparison is illustrated in [the theoretical model comparison figure](figures/fig-4-9-theoretical-experimental-comparison.png).

## 4. Original Research Findings

The original research reported the following results:

| Component | Reported result |
|---|---|
| CNN classification | Approximately 96% test accuracy |
| Upper-left coordinate regression | Approximately 96% |
| Upper-right coordinate regression | Approximately 96% |
| Bottom coordinate regression | Approximately 93% |
| Left coordinate regression | Approximately 93% |
| Right coordinate regression | Approximately 93% |
| Theoretical-to-experimental curve agreement | Approximately 90% |

The coordinate-regression accuracy was defined using a two-pixel tolerance between predicted and ground-truth coordinates.

The approximately 90% theoretical-to-experimental agreement was the original research's assessment of the representative displacement-curve comparison. It was not presented as a standardized statistical goodness-of-fit metric.

The original research concluded that:

- CNN-based detection produced clearer displacement curves with less visible noise than the traditional image-processing method.
- The four-stage theoretical model described the main characteristics of the droplet-transfer cycle.
- The theoretical displacement trajectory showed substantial agreement with the representative CNN-derived experimental trajectory.

## 5. Reconstructed Repository

This repository preserves and modernizes the computational components of the research.

### 5.1 Deep-learning Reconstruction

The original TensorFlow-era implementation has been reconstructed using:

- Python 3.11.
- PyTorch.
- OpenCV.
- NumPy.

The reconstructed module includes frame preprocessing, sample selection, annotation tools, dataset construction, CNN training, inference, and result visualization.

The reconstructed classification dataset contains 16,000 image-window samples:

- 12,000 training samples.
- 4,000 test samples.

The five reconstructed regression datasets are maintained separately for the five boundary targets. Their sample counts and evaluation results are documented in [Model Specification](model_specification.md).

### 5.2 Reconstructed Detection Results

The current deep-learning implementation produces detection CSV files, annotated GIFs, and boundary displacement plots for seven video sequences:

`20`, `21`, `22`, `24c`, `26c`, `29`, and `33`.

The current traditional-vision result set contains four video sequences:

`20`, `24c`, `26c`, and `33`.

The difference in coverage reflects the available reconstructed outputs. It should not be interpreted as evidence that the original research used only four videos for traditional image processing.

### 5.3 Reconstructed Model Results

The reconstructed PyTorch training results are distinct from the historical metrics reported in the original research.

The reconstructed classifier achieved 98.50% test accuracy in the recorded training run. The five regression models achieved different test MAE and within-two-pixel accuracy values, depending on the target and test-set size.

These results are documented separately in [Model Specification](model_specification.md).

### 5.4 Theoretical Model in the Reconstructed Project

The original theoretical model is retained as a physical reference for interpreting droplet-bottom displacement curves.

The historical piecewise displacement function uses four time intervals and the parameter values derived in the original analysis. Its initial time and displacement can be adjusted to align the model with a selected experimental cycle.

The original approximately 90% agreement should remain associated with the historical comparison. The reconstructed repository does not claim that this agreement has been independently re-established for every video or for the current PyTorch outputs.

## 6. Scope and Limitations

The repository is intended to preserve the research workflow and provide a modern, inspectable implementation. It is not claimed to be a bit-for-bit reproduction of the original software environment.

Important limitations include:

- The reconstructed regression datasets use sample-based splitting. Samples from the same source frame may therefore appear in different subsets.
- The classification dataset uses source-frame-based grouping.
- Regression test-set sizes vary substantially between targets; metrics should be interpreted together with their sample counts.
- The archived traditional-vision C++ source has not been fully verified under OpenCV 5.
- The theoretical model comparison in the original research used a representative CNN-derived droplet-bottom displacement curve. It should not be generalized to all five boundary coordinates without further analysis.
- The reconstructed detection outputs and model checkpoints document the modern implementation, not the exact runtime environment used in the original experiments.

## 7. Conclusion

The research combined computer vision and theoretical modelling to investigate droplet-transfer dynamics in piezoelectric actuator-assisted GMAW.

The computational workflow extracts and samples experimental frames, annotates droplet regions and boundaries, trains CNN models, detects droplet coordinates, and generates displacement visualizations.

The theoretical model provides a physical description of the transfer cycle and a reference trajectory for comparison with the experimentally measured droplet-bottom motion.

The reconstructed repository preserves these connected components while distinguishing historical research results from the metrics of the modernized implementation.
