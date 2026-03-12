# Unscented Kalman Filter

Higher-accuracy nonlinear estimation[@Julier1996][@Wan2000]. Instead of linearizing with Jacobians, the UKF propagates a set of carefully chosen **sigma points** through the nonlinear dynamics and reconstructs the output distribution.

Captures mean and covariance to second order (vs. first order for EKF).

## Usage

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

For unicycle models:

```python
estimator = model.unicycle.estimator.ukf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

## When to Use

Same scenarios as EKF, but when you need better accuracy or when the EKF diverges. Slightly more expensive than EKF.

## API Reference

See the [model API reference](../../api/model.md) for full signatures.

\bibliography
