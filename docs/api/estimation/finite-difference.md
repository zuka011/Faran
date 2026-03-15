---
status: draft
---

# Finite Difference Estimator

Numerical differentiation of consecutive observations. Requires no tuning and produces no covariance — use a [Kalman-family estimator](kalman.md) when uncertainty is needed.

## Formulas

**Speed** (all models, $T \geq 2$): projection onto heading direction with signed magnitude:

$$
v_t = \operatorname{sign}\!\bigl(\Delta x_t \cos\theta_t + \Delta y_t \sin\theta_t\bigr) \; \frac{\|\Delta \mathbf{p}_t\|}{\Delta t}
$$

**Angular velocity** (unicycle, $T \geq 2$):

$$
\omega_t = \frac{\theta_t - \theta_{t-1}}{\Delta t}
$$

**Acceleration** (bicycle, $T \geq 3$):

$$
a_t = \frac{v_t - v_{t-1}}{\Delta t}
$$

**Steering angle** (bicycle, $T \geq 2$): derived from heading rate $\dot\theta_t$ and kinematic bicycle geometry:

$$
\delta_t = \arctan\!\left(\frac{L \, \dot\theta_t}{\sqrt{v_t^2 - l_r^2 \, \dot\theta_t^2}}\right)
$$

Set to zero when $|v_t| < \varepsilon$ or $v_t^2 \leq l_r^2 \dot\theta_t^2$.

## Bicycle

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | $\Delta t$ (seconds) |
| `wheelbase` | `float` | — | Axle-to-axle distance $L$ |
| `rear_axle_distance` | `float` | `0.0` | Reference point to rear axle $l_r$ |

```python
from faran.numpy import model

estimator = model.bicycle.estimator.finite_difference(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
)
```

::: faran.models.bicycle.basic.NumPyFiniteDifferenceBicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.bicycle.accelerated.JaxFiniteDifferenceBicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Unicycle

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | $\Delta t$ (seconds) |

```python
estimator = model.unicycle.estimator.finite_difference(time_step_size=0.1)
```

::: faran.models.unicycle.basic.NumPyFiniteDifferenceUnicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.unicycle.accelerated.JaxFiniteDifferenceUnicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Integrator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | $\Delta t$ (seconds) |

$$
\mathbf{v}_t = \frac{\mathbf{x}_t - \mathbf{x}_{t-1}}{\Delta t}
$$

```python
estimator = model.integrator.estimator.finite_difference(time_step_size=0.1)
```

::: faran.models.integrator.basic.NumPyFiniteDifferenceIntegratorStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.integrator.accelerated.JaxFiniteDifferenceIntegratorStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
