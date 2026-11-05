# Original Research Pipeline

## 1. Overview

The original research combined experimental image analysis and theoretical modelling to investigate droplet-transfer dynamics in piezoelectric actuator-assisted Gas Metal Arc Welding (GMAW).

Its workflow consisted of three major parts:

1. Traditional image processing for droplet-boundary detection.
2. CNN-based classification and coordinate regression for droplet detection.
3. Theoretical modelling of the droplet-transfer process and comparison with experimental displacement curves.

The deep-learning approach was the principal image-based detection method for the theoretical-model comparison. Traditional image processing was developed as a supplementary method for comparison.

This document describes the original research workflow and distinguishes it from the modern implementation maintained in this repository.

## 2. Experimental Image Characteristics

High-speed cameras were used to record droplet transfer under different experimental conditions.

The images were affected by intense arc-light illumination. This caused several difficulties:

- Redundant contours caused by arc-light reflection and refraction.
- Missing or blurred sections of the droplet boundary.
- Fragmented contours around the droplet and weld pool.

These characteristics made it difficult to obtain reliable droplet coordinates using fixed image-processing rules.

## 3. Traditional Image-Processing Pipeline

The original traditional-vision algorithm was implemented in C++ using OpenCV.

### 3.1 Image Preprocessing

The original pipeline applied:

1. Grayscale conversion.
2. Gaussian filtering using a 5 × 5 kernel.
3. Canny edge detection with an empirically selected threshold of 50.

The output was a binary edge image containing the wire, droplet, weld pool, and other visible contours.

### 3.2 Contour Reconstruction

The original research identified three major contour problems:

- Redundant internal contours.
- Missing sections of the droplet boundary.
- Small fragmented contours.

Contour reconstruction consisted of two main procedures.

**Contour tracing and redrawing**

The algorithm traced existing contour pixels and reconstructed single-pixel paths. It sampled points along the paths and filtered short contour fragments.

**Contour extension**

The algorithm estimated the direction of a contour near its endpoint and extended it into a bounded angular region. If the extension reached another contour endpoint, the two contours were connected to approximate a continuous boundary.

The reconstructed contour was represented as an ordered sequence of coordinates.

### 3.3 Boundary Detection

The original traditional-vision method detected three boundary positions:

- Upper-left boundary.
- Upper-right boundary.
- Bottom boundary.

The upper boundary points were obtained by examining changes in the direction of the reconstructed contour. The bottom boundary was obtained from the lowest point on the contour.

### 3.4 Original Limitations

The original research reported that the method required substantial manual parameter selection, including filtering, edge detection, contour sampling, and extension parameters.

Because image quality varied under arc-light interference, the reconstructed contours were not consistently reliable. The resulting displacement curves contained substantial noise and could lose important motion characteristics.

The traditional method was therefore used as a supplementary comparison rather than the primary source of displacement measurements.

## 4. Original Deep-learning Pipeline

The original deep-learning implementation used Python, OpenCV, NumPy, and TensorFlow.

### 4.1 Video Frame Preparation

The original research used seven video sequences.

The original frame resolution was 500 × 352 pixels. Frames were reduced to 40% of their original dimensions, resulting in 200 × 140 images.

A total of 14,074 frames were available across the seven sequences.

### 4.2 Sample Selection

The original sampling strategy aimed to preserve diversity across experimental conditions and droplet-transfer cycles.

The strategy considered:

1. Selecting among the seven video sequences with equal probability.
2. Selecting among the cycles within a selected sequence.
3. Sampling more frequently during droplet detachment and falling, and less frequently while the droplet remained attached to the wire.

A total of 4,000 frames were selected to form the aggregate image set.

### 4.3 Classification Dataset

Four 32 × 32 image windows were manually selected from each of the 4,000 sampled frames:

- No droplet.
- Complete droplet.
- Upper droplet region.
- Lower droplet region.

This produced 16,000 classification samples, with 4,000 samples per class.

The original research used 3,000 training samples and 1,000 test samples per class.

### 4.4 Regression Datasets

The regression task used image windows containing relevant droplet regions.

Five boundary coordinates were annotated:

- Upper-left.
- Upper-right.
- Bottom.
- Left.
- Right.

The original research selected 12,000 image windows for regression dataset construction. Since not every window contained every boundary point, five target-specific datasets were produced with different sample counts.

| Target      | Total samples | Training | Test |
|-------------|--------------:|---------:|-----:|
| Upper-left  |         9,600 |    9,200 |  400 |
| Upper-right |         9,100 |    9,000 |  100 |
| Bottom      |         7,920 |    7,680 |  240 |
| Left        |        10,960 |   10,560 |  400 |
| Right       |        10,960 |   10,880 |   80 |

### 4.5 CNN Training

Both networks were based on a compact LeNet-style architecture.

The classification network predicted one of four window classes. The regression network predicted a two-dimensional coordinate.

The original research specified:

- Adam optimizer.
- Learning rate of 0.001.
- Classification batch size of 100.
- Regression batch size of 80.
- Cross-entropy loss for classification.
- L2 loss for coordinate regression.

The original research reported approximately 96% classification accuracy. Coordinate predictions were reported as approximately 96% for the upper corners and 93% for the bottom and side boundaries, using a two-pixel tolerance.

### 4.6 Full-frame Detection

The original detector applied a 32 × 32 sliding window to the resized full frame.

The classification network first identified windows containing droplet regions. The regression network then predicted boundary coordinates within candidate windows.

Local coordinates were converted back to the full-frame coordinate system. Multiple predictions for the same boundary were consolidated to obtain more stable estimates.

The final displacement curves were generated from the detected boundary positions across the video sequence.

## 5. Theoretical Dynamic Modelling

### 5.1 Physical Process

The theoretical analysis describes one complete droplet-transfer cycle as four stages.

**Stage 1: Droplet dwelling**

The wire melting speed \(v_m\) is equal to the wire feeding speed \(v_f\):

\[
v_m=v_f
\]

The droplet remains at an approximately constant position. If its initial displacement is \(x_0\), then:

\[
x(t)=x_0
\]

**Stage 2: Wire energy storage**

The piezoelectric actuator clamps the wire tip. The wire tip stops advancing while the wire continues to be fed, producing elastic deformation.

The wire feeding displacement during the clamping period is:

\[
\lambda=v_f t_f
\]

where \(t_f\) is the clamping duration.

During this stage, the droplet continues moving upward due to wire melting:

\[
x(t)=x_0-v_m(t-t_0-t_1)
\]

The bent wire is further analysed using an equivalent compressed-rod model. Its deflection is represented by:

\[
w=A\sin(kx)
\]

where:

\[
k=\sqrt{\frac{F}{EI}}
\]

Here, \(F\) is the equivalent axial force, \(E\) is Young's modulus, and \(I\) is the cross-sectional second moment of area.

The analysis derives the deflection amplitude from the wire feeding displacement and the arc-shaped wire geometry. The resulting expression describes the wire's deflection during the energy-storage stage.

### 5.2 Wire Rebound

When the piezoelectric actuator releases the wire, the stored elastic energy drives the wire back toward its original straight configuration.

The deformed wire is approximated as an equivalent spring-mass system.

The equivalent vibrating mass is:

\[
m_w=\frac{1}{3}m_0+m_r
\]

where \(m_0\) and \(m_r\) are the masses of the equivalent spring section and the remaining wire section, respectively.

The equivalent spring stiffness is:

\[
k_w=\frac{3EI}{L_0^3}
\]

The wire rebound is modelled as a quarter-cycle simple harmonic motion. The corresponding droplet displacement is:

\[
x(t)=x_0-\lambda
\sin\left(
\sqrt{\frac{k_w}{m_w}}
(t-t_0-t_1-t_2)+\frac{\pi}{2}
\right)
\]

The duration of this stage is:

\[
t_w=\frac{\pi}{2}\sqrt{\frac{m_w}{k_w}}
\]

### 5.3 Droplet Detachment

After the wire returns to its original length, its tip rapidly decelerates. The droplet continues moving due to inertia.

The droplet's connection to the wire becomes progressively smaller until the droplet overcomes the surface tension of the liquid connection and detaches.

The droplet is approximated as an equivalent spring-mass system with stiffness:

\[
k_d=7.74+127961.29m_d
\]

where \(m_d\) is the droplet mass.

The droplet mass is estimated by treating the droplet as a sphere:

\[
m_d=\rho_d V_d
=\rho_d\frac{\pi d_d^3}{6}
\]

The detachment displacement amplitude \(h_d\) is derived from the energy relationship:

\[
\frac{1}{2}m_dv^2+m_dgh_d
=
\frac{1}{2}k_dh_d^2
\]

The original analysis obtained an approximate initial velocity of \(0.38\ \mathrm{m/s}\) for this stage and a displacement amplitude of:

\[
h_d=0.0015\ \mathrm{m}
\]

The resulting droplet-detachment displacement is:

\[
\begin{aligned}
x(t)={}&x_0-\lambda
\sin\left(
\sqrt{\frac{k_w}{m_w}}t_w+\frac{\pi}{2}
\right)\\
&+h_d\sin\left(
\sqrt{\frac{k_d}{m_d}}
(t-t_0-t_1-t_2-t_w)
\right)
\end{aligned}
\]

The duration of this stage is:

\[
t_d=\frac{\pi}{2}\sqrt{\frac{m_d}{k_d}}
\]

The model treats detachment as the end of the corresponding transfer cycle.

### 5.4 Complete Piecewise Displacement Model

The four stage-specific expressions are combined into one piecewise displacement function.

After substituting the experimental parameters from the original analysis, the theoretical displacement becomes:

\[
x(t)=
\begin{cases}
x_0,
& t_0<t\leq t_0+0.02\\[4pt]
x_0-0.05(t-t_0-0.02),
& t_0+0.02<t\leq t_0+0.04\\[4pt]
x_0-0.001\sin\left(
380.6(t-t_0-0.04)+\frac{\pi}{2}
\right),
& t_0+0.04<t\leq t_0+0.0441\\[4pt]
x_0+0.0015\sin\left(
811.6(t-t_0-0.0441)
\right),
& t_0+0.0441<t\leq t_0+0.0461
\end{cases}
\]

The displacement is expressed in metres and time in seconds.

The parameters \(x_0\) and \(t_0\) define the initial displacement and time. They can be adjusted to align the theoretical curve with a selected experimental transfer cycle.

### 5.5 Model Interpretation

The first two stages are directly controlled by the wire-feeding system and piezoelectric actuator. The latter two stages describe the resulting rebound and droplet-detachment motion after the actuator releases the wire.

The original analysis estimated that the period during which droplet motion was no longer directly controlled by the experimental system accounted for approximately 15% of the directly controlled duration.

This provides a physical interpretation of how the actuator controls most of the droplet-transfer cycle while the final detachment is driven by the stored elastic energy and droplet inertia.

## 6. Theoretical Model and Experimental Comparison

The original research compared the theoretical piecewise displacement function with a representative droplet-bottom displacement curve obtained using the CNN-based detection method.

The comparison procedure was:

1. Detect the droplet bottom in each experimental frame.
2. Generate a displacement-versus-time curve.
3. Select a representative transfer cycle.
4. Determine the initial time and displacement for that cycle.
5. Substitute the initial conditions into the theoretical model.
6. Compare the theoretical and experimental trajectories.

For the representative comparison, the original analysis selected:

\[
x_0=4.6\times10^{-3}\ \mathrm{m}
\]

\[
t_0=0.094\ \mathrm{s}
\]

The resulting theoretical curve was compared with the experimentally detected droplet-bottom trajectory.

The original research reported approximately 90% overall agreement. It also discussed possible deviations caused by variations in wire feeding speed and wire melting speed during the experiment.

The historical comparison figure is referenced in:

[`docs/figures/theoretical_model_comparison.png`](figures/theoretical_model_comparison.png)

**Scope of the comparison:** The original theoretical comparison used a CNN-derived droplet-bottom trajectory. It did not independently validate all five CNN regression targets or all traditional-vision boundary curves against the theoretical model.

## 7. Relationship to the Reconstructed Repository

The current repository modernizes and documents the computational components of the original workflow.

| Component                    | Original research                            | Current repository                                                     |
|------------------------------|----------------------------------------------|------------------------------------------------------------------------|
| Traditional image processing | C++ and OpenCV                               | Preserved C++ source; modern OpenCV 5 compatibility not fully verified |
| Deep-learning framework      | TensorFlow                                   | PyTorch                                                                |
| Image preprocessing          | Frames reduced to 40%                        | Reconstructed preprocessing workflow                                   |
| Classification               | Four classes, 32 × 32 windows                | Four classes, 32 × 32 windows                                          |
| Regression                   | Five boundary targets                        | Five independent regression models                                     |
| Model training               | Adam, learning rate 0.001                    | Adam, learning rate 0.001, reconstructed training runs                 |
| Detection                    | Sliding-window classification and regression | Reconstructed sliding-window detector                                  |
| Theoretical model            | Four-stage physical model                    | Documented as a reference for interpreting displacement curves         |
| Results                      | Historical thesis figures and metrics        | CSV files, GIFs, plots, and model checkpoints                          |

The reconstructed implementation should not be interpreted as an exact reproduction of the original TensorFlow runtime. Changes in framework, data handling, training execution, and inference implementation are documented in the other specification documents.
