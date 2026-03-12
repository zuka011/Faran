# Unicycle Model

Three state variables with direct velocity control[@Oriolo2002]. Suitable for differential-drive or omnidirectional robots.

## Equations

$$
x_{t+1} = x_t + v_t \cos(\theta_t) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t) \, \Delta t, \quad
\theta_{t+1} = \theta_t + \omega_t \, \Delta t
$$

| Component | Variables | Meaning |
|-----------|-----------|---------|
| State | $[x, y, \theta]$ | Position, heading |
| Controls | $[v, \omega]$ | Linear velocity, angular velocity |

## Usage

```python
from faran.numpy import model

unicycle = model.unicycle.dynamical(
    time_step_size=0.1,
    speed_limits=(0.0, 5.0),
    angular_velocity_limits=(-1.0, 1.0),
)
```

## When to Use

Robots or agents where you control velocity and turning rate directly, or when you don't need a detailed vehicle model.

## API Reference

See the [model API reference](../../api/model.md) for full signatures.

\bibliography
