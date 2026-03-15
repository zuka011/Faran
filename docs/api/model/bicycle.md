---
status: draft
---

# Kinematic Bicycle Model

Four-state model representing a wheeled vehicle with acceleration and steering control[@Kong2015]. Discretized via Euler integration.

## Dynamics

The slip angle $\beta$ at the reference point:

$$
\beta = \arctan\!\left(\frac{l_r}{L} \tan(\delta)\right)
$$

State transition ($\Delta t$ = time step):

$$
x_{t+1} = x_t + v_t \cos(\theta_t + \beta) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t + \beta) \, \Delta t
$$

$$
\theta_{t+1} = \theta_t + \frac{v_t}{L} \cos(\beta) \tan(\delta_t) \, \Delta t, \quad
v_{t+1} = v_t + a_t \, \Delta t
$$

When $l_r = 0$, $\beta = 0$ and the equations reduce to the standard rear-axle model.

| Component | Variables | Description |
|-----------|-----------|-------------|
| State | $[x, y, \theta, v]$ | Position, heading, speed |
| Controls | $[a, \delta]$ | Acceleration, steering angle |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | Integration step $\Delta t$ (seconds) |
| `wheelbase` | `float` | `1.0` | Distance between axles $L$ |
| `rear_axle_distance` | `float` | `0.0` | Distance from reference point to rear axle $l_r$ |
| `speed_limits` | `tuple[float, float] \| None` | `None` | Clamp bounds on $v$ |
| `steering_limits` | `tuple[float, float] \| None` | `None` | Clamp bounds on $\delta$ |
| `acceleration_limits` | `tuple[float, float] \| None` | `None` | Clamp bounds on $a$ |

```python
from faran.numpy import model

bicycle = model.bicycle.dynamical(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
    speed_limits=(0.0, 15.0), steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

::: faran.models.bicycle.basic.NumPyBicycleModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

::: faran.models.bicycle.accelerated.JaxBicycleModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

## Obstacle Model

Propagates obstacle states forward using bicycle kinematics. Paired with a [state estimator](../estimation/index.md) and [predictor](../predictor/index.md) for obstacle tracking.

The obstacle model follows the CSAA (Constant Steering Angle & Acceleration) assumption[@Schubert2008]: state $\mathsf{x} = [x, y, \theta, v, a, \delta]$ with $a$ and $\delta$ held constant over the prediction horizon.

::: faran.models.bicycle.basic.NumPyBicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.bicycle.accelerated.JaxBicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

\bibliography
