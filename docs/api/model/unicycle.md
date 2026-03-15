---
status: draft
---

# Unicycle Model

Three-state model representing a point robot with direct velocity and angular velocity control[@Oriolo2002]. Discretized via Euler integration.

## Dynamics

$$
x_{t+1} = x_t + v_t \cos(\theta_t) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t) \, \Delta t, \quad
\theta_{t+1} = \theta_t + \omega_t \, \Delta t
$$

| Component | Variables | Description |
|-----------|-----------|-------------|
| State | $[x, y, \theta]$ | Position, heading |
| Controls | $[v, \omega]$ | Linear velocity, angular velocity |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | Integration step $\Delta t$ (seconds) |
| `speed_limits` | `tuple[float, float] \| None` | `None` | Clamp bounds on $v$ |
| `angular_velocity_limits` | `tuple[float, float] \| None` | `None` | Clamp bounds on $\omega$ |

```python
from faran.numpy import model

unicycle = model.unicycle.dynamical(
    time_step_size=0.1,
    speed_limits=(-5.0, 5.0),
    angular_velocity_limits=(-1.0, 1.0),
)
```

::: faran.models.unicycle.basic.NumPyUnicycleModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

::: faran.models.unicycle.accelerated.JaxUnicycleModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

## Obstacle Model

Propagates obstacle states forward using unicycle kinematics. Follows the CTRV (Constant Turn Rate & Velocity) assumption[@Schubert2008]: state $\mathsf{x} = [x, y, \theta, v, \omega]$ with $v$ and $\omega$ held constant over the prediction horizon.

::: faran.models.unicycle.basic.NumPyUnicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.unicycle.accelerated.JaxUnicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

\bibliography
