---
status: draft
---

# model

Dynamics models define how the system state evolves given a control input. The [MPPI](../mppi/index.md) planner uses the model to simulate forward rollouts.

All models perform Euler integration with a fixed time step $\Delta t$.

## Available Models

| Model | State $\mathsf{x}$ | Controls $\mathsf{u}$ | Parameters |
|-------|---------------------|------------------------|------------|
| [Bicycle](bicycle.md) | $[x, y, \theta, v]$ | $[a, \delta]$ | Wheelbase $L$, rear axle distance $l_r$ |
| [Unicycle](unicycle.md) | $[x, y, \theta]$ | $[v, \omega]$ | — |
| [Integrator](integrator.md) | $[x_1, \ldots, x_n]$ | $[u_1, \ldots, u_n]$ | — |

## State and Input Limits

All models accept optional limit tuples that clamp state and control values after each integration step. Omit a limit to leave that dimension unconstrained.

## Obstacle Models

Each dynamics model has a corresponding obstacle variant for [motion prediction](../obstacles/index.md). These propagate observed obstacle states forward using the same kinematics, paired with a [state estimator](../estimation/index.md):

```python
from faran.numpy import model

obstacle_model = model.bicycle.obstacle(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
)
```

## DynamicalModel Protocol

::: faran.types.DynamicalModel
    options:
      show_root_heading: true
      heading_level: 3
