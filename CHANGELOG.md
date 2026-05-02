# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] — 2026-03-21

### Changed

- Minor readability improvements in the documentation.
- A preference cost component that can be used to incorporate arbitrary heuristics for trajectory selection.

### Fixed

- Fixed reduction on non-aligned axes causing XLA compilation to be very long.
- Overly strict typing for obstacle state covariances.
- Hungarian obstacle assignment unable to handle NaN values in the obstacles states for the current time step.

## [0.3.0] — 2026-03-15

### Added

- `py.typed` markers for PEP 561 compliance
- Static factory methods for noise covariance bounds

### Changed

- Refined trajectory and sampler APIs for consistency
- Improved additional plot responsiveness in the visualizer
- Partially reworked documentation structure

## [0.2.10] — 2026-03-13

### Changed

- Improved noise covariance clamping logic; added support for specifying a noise ceiling

## [0.2.9] — 2026-03-13

### Added

- Visualizer support for rendering noisy obstacle observations

## [0.2.8] — 2026-03-12

### Fixed

- Adaptive noise filter crashing when obstacle states are missing

## [0.2.7] — 2026-03-12

### Changed

- Moved `NoisyObstacleStateObserver` construction into the factory namespace

## [0.2.6] — 2026-03-11

### Added

- `rear_axle_distance` parameter to kinematic bicycle model for arbitrary reference points

### Fixed

- Collision cost `weight` is now applied after risk metric evaluation, not inside it

## [0.2.5] — 2026-02-22

### Added

- Heading orientation support in the Hungarian algorithm obstacle ID assignment

## [0.2.4] — 2026-02-22

### Changed

- JAX Kalman filter update step switched to Joseph form for better numerical stability

## [0.2.3] — 2026-02-21

### Changed

- Project renamed to **Faran**

## [0.2.2] — 2026-02-19

### Added

- Test suite for visualizer geometry computations
- Automatic Jupyter notebook generation and upload in CI

## [0.2.1] — 2026-02-18

### Added

- State estimators: Kalman Filter, Extended KF, and Unscented KF for obstacle tracking
- Finite difference state estimators for noise-free scenarios
- Adaptive process and observation noise estimation (Mohamed & Schwarz 1999)

### Changed

- Reworked obstacle motion prediction API; estimator and model are now separate

## [0.2.0] — 2026-02-04

### Added

- Unicycle dynamics model (3-state: x, y, θ) with speed and angular velocity inputs
- Unicycle obstacle model for motion prediction
- Highway scenario integration test

### Changed

- Redesigned obstacle state and pose API for clarity

## [0.1.15] — 2026-01-17

### Added

- SAT (Separating Axis Theorem) based polygon distance extractor
- Halton quasi-random spline sampler for temporally smooth, low-discrepancy perturbations
- Piecewise fixed-width boundary with arc-length breakpoints

## [0.1.14] — 2026-01-14

### Added

- Task completion metric (goal reached, time, stretch ratio, progress fraction)
- Constraint violation metric (boundary and input limit violations)
- Comfort metric (jerk, lateral acceleration, smoothness)

## [0.1.11] — 2026-01-12

### Added

- Fixed-width corridor boundary cost with symmetric and asymmetric half-widths
- Boundary trace in the visualizer

## [0.1.6] — 2026-01-05

### Added

- Control effort cost (input magnitude penalty)
- Additional risk metrics from RisKit: Mean-variance, VaR, CVaR, Entropic risk

## [0.1.0] — 2026-01-03

### Added

- Initial release of Faran
- MPPI planner with NumPy and JAX backends (CPU and GPU)
- MPCC formulation: contouring, lag, and progress costs
- Kinematic bicycle model and single integrator model
- Gaussian perturbation sampler
- Collision cost with circle-to-circle distance
- Waypoint (B-spline) and line reference trajectory types
- Running obstacle history with ID tracking via Hungarian algorithm
- MPCC error and collision evaluation metrics
- Standalone HTML visualizer (`faran-visualizer`) with interactive Plotly plots

[0.3.0]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.10...v0.3.0
[0.2.10]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.9...v0.2.10
[0.2.9]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.8...v0.2.9
[0.2.8]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.7...v0.2.8
[0.2.7]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.6...v0.2.7
[0.2.6]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.5...v0.2.6
[0.2.5]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.4...v0.2.5
[0.2.4]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.3...v0.2.4
[0.2.3]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.2...v0.2.3
[0.2.2]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.1...v0.2.2
[0.2.1]: https://gitlab.com/risk-metrics/faran/-/compare/v0.2.0...v0.2.1
[0.2.0]: https://gitlab.com/risk-metrics/faran/-/compare/v0.1.17...v0.2.0
[0.1.15]: https://gitlab.com/risk-metrics/faran/-/compare/v0.1.11...v0.1.15
[0.1.14]: https://gitlab.com/risk-metrics/faran/-/compare/v0.1.11...v0.1.14
[0.1.11]: https://gitlab.com/risk-metrics/faran/-/compare/v0.1.6...v0.1.11
[0.1.6]: https://gitlab.com/risk-metrics/faran/-/compare/v0.1.0...v0.1.6
[0.1.0]: https://gitlab.com/risk-metrics/faran/-/tags/v0.1.0
