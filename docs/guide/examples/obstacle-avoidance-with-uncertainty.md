---
reading_time: true
status: draft
---

# Obstacle Avoidance with Uncertainty

Moving obstacles make things harder — the planner doesn't just need to avoid where obstacles *are*, but where they *might be*. This example swaps in an [EKF estimator](../../api/estimation/index.md) for covariance propagation and a [mean-variance risk metric](../../api/costs/risk.md) that makes collision avoidance uncertainty-aware.

## Reference Path

A longer path than before (`path_length=100.0`, 8 waypoints) — the dynamic obstacles need more space to interact with the planner:

```python
--8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:reference"
```

## Setup

```python
--8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:setup"
```

The changes from [Obstacle Avoidance](obstacle-avoidance.md):

| Component                                                | What changed                                                                                                                                           |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `create_obstacles.dynamic(...)`                          | Four obstacles with constant velocities — three moving, one stationary                                                                                 |
| `model.bicycle.estimator.ekf(...)`                       | [Extended Kalman Filter](../../api/estimation/index.md) replaces finite-difference estimation — propagates covariance through the bicycle motion model |
| `risk.mean_variance(gamma=0.5, sample_count=10)`         | [Mean-variance risk metric](../../api/costs/risk.md) penalizes both expected collision cost and its variance                                           |
| `metric=risk_collector` on `costs.safety.collision(...)` | Collision cost evaluates risk over 10 sampled obstacle poses drawn from the predicted distribution, not just the mean prediction                       |

The EKF propagates covariance through the motion model at each prediction step. Its `process_noise_covariance` (1e-4) and `observation_noise_covariance` (1e-8) control the trust balance between the model and observations — low observation noise means the filter trusts observations heavily, while moderate process noise allows the model prediction to drift. The observation history was increased from 2 to 10 time steps (`horizon=10` in `obstacle_states_running_history`) to give the EKF more data for state estimation.

The mean-variance metric converts the sampled collision costs into a single scalar: $\mathbb{E}[c] + \gamma \cdot \text{Var}(c)$. Higher `gamma` makes the planner more conservative — it penalizes *uncertain* near-misses more than *confident* near-misses. With `gamma=0.5`, the planner gives moderate weight to prediction uncertainty.

## Simulation Loop

Identical to [Obstacle Avoidance](obstacle-avoidance.md) — the risk-aware behavior is entirely in the cost function:

```python
--8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:loop"
```

To run:

```python
components = create()
result = run(*components)
```

## Result

```python
--8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:visualize"
```

```python
import asyncio

asyncio.run(visualize(result))
```

The visualization shows four obstacles — three moving with constant velocities of up to 2.5 m/s and one stationary — along a 100 m reference path with a 3.0 m corridor:

<iframe src="../../../visualizations/mpcc-simulation/doc-dynamic-obstacles-uncertain.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full example"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py"
    ```

You can also run this example on Binder → [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zurabmu/faran/main?filepath=notebooks/04_obstacle_avoidance_with_uncertainty.ipynb){ .binder-badge }

## Next Steps

- **[Risk Metrics](../../api/costs/risk.md)** — Available risk metrics for uncertainty-aware cost evaluation.
- **[State Estimation](../../api/estimation/index.md)** — KF, EKF, and UKF estimators with adaptive noise.
- **[MPPI Planning](../../api/mppi/index.md)** — Customize the planner with different models, samplers, and costs.
