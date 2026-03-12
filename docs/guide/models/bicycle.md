# Kinematic Bicycle Model

Four state variables and two control inputs, discretized via Euler integration[@Polack2017]. Supports an optional **rear axle distance** $l_r$ that shifts the reference point from the rear axle toward the center of gravity.

## Equations

$$
\beta = \arctan\!\left(\frac{l_r}{L} \tan(\delta_t)\right)
$$

$$
x_{t+1} = x_t + v_t \cos(\theta_t + \beta) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t + \beta) \, \Delta t
$$

$$
\theta_{t+1} = \theta_t + \frac{v_t}{L} \cos(\beta) \tan(\delta_t) \, \Delta t, \quad
v_{t+1} = v_t + a_t \, \Delta t
$$

When $l_r = 0$ (default), the reference point is at the rear axle and $\beta = 0$.

| Component | Variables | Meaning |
|-----------|-----------|---------|
| State | $[x, y, \theta, v]$ | Position, heading, speed |
| Controls | $[a, \delta]$ | Acceleration, steering angle |
| Parameters | $L$, $l_r$ | Wheelbase, rear axle distance |

## Usage

```python
from faran.numpy import model

bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

To place the reference point at the center of gravity:

```python
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    rear_axle_distance=1.0,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

## When to Use

Vehicles with front-axle steering (cars, trucks). The bicycle approximation is valid at moderate speeds and steering angles where tire slip is negligible. Use `rear_axle_distance` when the tracked position is not the rear axle (e.g., a GPS antenna or center of gravity).

## API Reference

See the [model API reference](../../api/model.md) for full signatures.

\bibliography
