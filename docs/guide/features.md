# Feature Overview

Everything listed below is implemented, tested, and available for both the NumPy and JAX backends, unless noted otherwise.

---

## :material-robot: Planners

<div class="grid cards" markdown>

-   **MPPI** · Model Predictive Path Integral

    ---

    An MPC algorithm that uses sampling and importance weighting to find optimal control sequences. Highly parallelizable and can solve many different MPC formulations, since it does not require gradients or convexity.

    Three configuration levels: [`mppi.base`](../api/mppi/index.md) (for any MPC problem), [`mppi.augmented`](../api/mppi/index.md) (If additional virtual states are needed), [`mppi.mpcc`](../api/mppi/index.md) (An extensible MPCC configuration).

    [:octicons-arrow-right-24: MPPI guide](../api/mppi/index.md#mppi)

</div>

---

## :material-format-list-checks: MPC Formulations

<div class="grid cards" markdown>

-   **MPCC** · Model Predictive Contouring Control

    ---

    MPC formulation that decomposes tracking error into contouring (lateral) and lag (longitudinal) components, with a virtual path parameter driving progress.

    [:octicons-arrow-right-24: Cost design](../api/costs/index.md) ·
    [:octicons-arrow-right-24: Concepts](concepts/mpcc.md)

</div>

---

## :material-car-side: Dynamics Models

<div class="grid cards" markdown>

-   **Kinematic Bicycle**

    ---

    4-state model ($x, y, \theta, v$) with acceleration and steering inputs.

    [:octicons-arrow-right-24: Bicycle Model](../api/model/bicycle.md)

-   **Unicycle**

    ---

    3-state model ($x, y, \theta$) with speed and angular velocity inputs.

    [:octicons-arrow-right-24: Unicycle Model](../api/model/unicycle.md)

-   **Integrator**

    ---

    Generic $n$-dimensional single integrator.

    [:octicons-arrow-right-24: Single Integrator](../api/model/integrator.md)

</div>

---

## :material-dice-multiple: Samplers

<div class="grid cards" markdown>

-   **Gaussian**

    ---

    Zero-mean Gaussian perturbations around a nominal control sequence.

    [:octicons-arrow-right-24: Gaussian Sampler](../api/sampler/gaussian.md)

-   **Halton + Spline**

    ---

    Halton quasi-random sequences mapped to a Gaussian distribution and interpolated with cubic splines. Temporally smooth, low-discrepancy perturbations.

    [:octicons-arrow-right-24: Halton Spline Sampler](../api/sampler/halton.md)

</div>

---

## :material-function-variant: Cost Functions

<div class="grid cards" markdown>

-   **Tracking**

    ---

    - **Contouring** — lateral deviation from the reference path
    - **Lag** — longitudinal deviation from the reference point
    - **Progress** — rewards forward motion along the path

    [:octicons-arrow-right-24: Tracking costs](../api/costs/tracking.md)

-   **Safety**

    ---

    - **Collision** — proximity to obstacles via signed distance
    - **Boundary** — states approaching corridor edges

    [:octicons-arrow-right-24: Safety costs](../api/costs/safety.md)

-   **Comfort**

    ---

    - **Control smoothing** — rate of change between consecutive inputs
    - **Control effort** — input magnitude

    [:octicons-arrow-right-24: Comfort costs](../api/costs/comfort.md)

</div>

---

## :material-shield-alert: Obstacle Avoidance

<div class="grid cards" markdown>

-   **State Estimation**

    ---

    - **Finite Difference Estimators** - for noise-free state estimation
    - **Kalman Filters** - for linear and nonlinear observation models
    - **Adaptive Noise Estimators** - for automatic process and observation noise tuning
  
    [:octicons-arrow-right-24: State Estimation](../api/estimation/index.md)

-   **Motion Prediction**

    ---

    All possible curvilinear prediction models for available system dynamics assumptions:

    - **Kinematic bicycle model** - e.g. for predicting traffic motion
    - **Kinematic unicycle model** - e.g. for mobile robots
    - **Single integrator model** - e.g. for pedestrians

    [:octicons-arrow-right-24: Motion Prediction](../api/predictor/index.md)

-   **Distance Computation & Collision Detection**

    ---

    - **Circle-to-circle** — distance between multi-circle approximations
    - **SAT** — distance between convex polygons approximation

    [:octicons-arrow-right-24: Distance Computation](../api/obstacles/index.md#distance-computation)

-   **Miscellaneous**

    ---

    - Running history for collecting observations with ID tracking
    - Hungarian algorithm for obstacle ID assignment
    - Synthetic noise for simulating noisy observations
    - Obstacle state sampling, e.g. for risk-aware motion planning

    [:octicons-arrow-right-24: Obstacle Avoidance](../api/obstacles/index.md)

</div>

---

## :material-chart-bell-curve: Risk Metrics

Risk-aware collision costs via the [riskit](https://gitlab.com/risk-metrics/riskit) library. The risk metric defines how a stochastic cost distribution is aggregated into a scalar cost for optimization. See [Risk Metrics](../api/costs/risk.md) for details.

| Metric             | Description                                      |
|--------------------|--------------------------------------------------|
| **Expected value** | Mean cost over samples                           |
| **Mean-variance**  | Mean + $\gamma \cdot$ variance                   |
| **VaR**            | Value at Risk at confidence $\alpha$             |
| **CVaR**           | Conditional Value at Risk at confidence $\alpha$ |
| **Entropic risk**  | Exponential risk measure with parameter $\theta$ |

---

## :material-road: Reference Trajectories & Boundaries

<div class="grid cards" markdown>

-   **Trajectories**

    ---

    - **Waypoints** — a B-spline path defined by a sequence of waypoints
    - **Line** — straight path between two endpoints

    [:octicons-arrow-right-24: Trajectories](../api/trajectory/index.md)

-   **Boundaries**

    ---

    - **Fixed-width corridor** — symmetric or asymmetric constant width
    - **Piecewise fixed-width** — segment-varying widths at arc-length breakpoints

    [:octicons-arrow-right-24: Boundaries](../api/boundary/index.md)

</div>

---

## :material-chart-line: Evaluation Metrics

Post-simulation evaluation for benchmarking and analysis.

| Metric                   | Measures                                                        |
|--------------------------|-----------------------------------------------------------------|
| **Collision**            | Minimum distances, collision detection per time step            |
| **MPCC error**           | Contouring and lag error over the trajectory                    |
| **Task completion**      | Goal reached, completion time, stretch ratio, progress fraction |
| **Constraint violation** | Boundary and limit violations                                   |
| **Comfort**              | Jerk, lateral acceleration, smoothness                          |

[:octicons-arrow-right-24: Metrics guide](../api/metrics/index.md) ·
[:octicons-arrow-right-24: API](../api/metrics/index.md)

---

## :material-map-marker-path: Roadmap

| Feature                                                   | Status  |
|-----------------------------------------------------------|---------|
| Additional planning algorithms (e.g. iLQR)                | Planned |
| Additional MPC formulations (e.g. Waypoint tracking)      | Planned |
| Components and application examples beyond mobile robots  | Planned |
| Plug-in system for the visualizer                         | Planned |
| More magic 🪄 in planner configuration (less boilerplate) | Planned |
| JIT friendlier architecture for data collection           | Planned |
