# Finite Difference Estimator

Estimates derivatives from consecutive observations. The simplest estimator — no tuning or noise assumptions required.

## Equations

Speed uses the displacement magnitude with direction from heading projection:

$$
v_t \approx \operatorname{sign}\!\big((x_t - x_{t-1})\cos\theta_t + (y_t - y_{t-1})\sin\theta_t\big) \,
\frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{\Delta t}
$$

Acceleration and steering angle:

$$
a_t = \frac{v_t - v_{t-1}}{\Delta t}, \quad
\delta_t \approx \arctan\!\left(\frac{L \, \dot\theta_t \operatorname{sign}(v_t)}{\sqrt{v_t^2 - l_r^2 \, \dot\theta_t^2}}\right)
$$

When $l_r = 0$, the steering formula simplifies to $\delta_t \approx \arctan(L \, \dot\theta_t / v_t)$.

## Usage

```python
from faran.numpy import model

estimator = model.bicycle.estimator.finite_difference(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
)
```

## Trade-offs

- **Pros:** No tuning, no noise assumptions, cheapest to compute.
- **Cons:** Very sensitive to observation noise. Does not produce covariance (no uncertainty estimates).

## API Reference

See the [model API reference](../../api/model.md) for full signatures.
