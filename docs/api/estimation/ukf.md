---
status: draft
---

# Unscented Kalman Filter

Second-order nonlinear state estimator for the [bicycle](../model/bicycle.md) and [unicycle](../model/unicycle.md) models. Propagates $2n + 1$ sigma points through the nonlinear dynamics and reconstructs the output distribution, capturing mean and covariance to second order (vs. first order for [EKF](ekf.md)).

## Sigma Point Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `1.0` | Spread of sigma points around the mean |
| `beta` | `float` | `2.0` | Prior knowledge weight (2 is optimal for Gaussian) |

Scaling parameter: $\lambda = \alpha^2 (n + \kappa) - n$ where $n$ is the state dimension and $\kappa = 0$.

## Bicycle

Same parameters as the [bicycle EKF](ekf.md#bicycle), plus `alpha` and `beta` on the underlying filter.

```python
from faran.numpy import model

estimator = model.bicycle.estimator.ukf(
    time_step_size=0.1,
    wheelbase=2.5,
    rear_axle_distance=1.0,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

::: faran.models.bicycle.basic.NumPyKfBicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - ukf

::: faran.models.bicycle.accelerated.JaxKfBicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - ukf

## Unicycle

Same parameters as the [unicycle EKF](ekf.md#unicycle), plus `alpha` and `beta` on the underlying filter.

```python
estimator = model.unicycle.estimator.ukf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

::: faran.models.unicycle.basic.NumPyKfUnicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - ukf

::: faran.models.unicycle.accelerated.JaxKfUnicycleStateEstimator
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - ukf
