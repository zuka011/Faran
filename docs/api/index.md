---
status: draft
---

# API Reference

Factory functions, protocols, and types for every Faran component.

## Modules

| Module | Purpose |
|--------|---------|
| [`mppi`](mppi/index.md) | MPPI planner configuration and execution |
| [`model`](model/index.md) | Dynamical models (bicycle, unicycle, integrator) |
| [`estimation`](estimation/index.md) | State estimation filters (KF, EKF, UKF, finite difference) |
| [`costs`](costs/index.md) | Cost functions (tracking, safety, comfort, risk) |
| [`sampler`](sampler/index.md) | Control input samplers (Gaussian, Halton) |
| [`trajectory`](trajectory/index.md) | Reference path definitions (line, waypoints) |
| [`boundary`](boundary/index.md) | Drivable corridor constraints |
| [`obstacles`](obstacles/index.md) | Obstacle state handling, distance, and sampling |
| [`predictor`](predictor/index.md) | Obstacle motion prediction |
| [`collectors`](collectors/index.md) | Simulation data collection |
| [`metrics`](metrics/index.md) | Post-simulation evaluation metrics |
| [`types`](types/index.md) | Protocols and type definitions |
| [`visualizer`](visualizer/index.md) | Interactive HTML visualizations |

## Backend Namespaces

All factory functions are accessed through backend namespaces:

```python
from faran.numpy import mppi, model, sampler, costs, trajectory, boundary, types
from faran.jax import mppi, model, sampler, costs, trajectory, boundary, types
```

Both namespaces expose identical APIs. See [Backends](../guide/backends.md).

## Conventions

| Signed Distance | Meaning |
|-----------------|---------|
| Positive | Inside valid region |
| Zero | On boundary |
| Negative | Violation |

State batches have shape $(T, D_x, M)$ where $T$ is the time horizon, $D_x$ the state dimension, and $M$ the number of rollouts. See [Conventions](../guide/concepts/conventions.md).
