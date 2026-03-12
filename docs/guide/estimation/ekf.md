# Extended Kalman Filter

For nonlinear models (bicycle, unicycle). Linearizes the dynamics around the current estimate using the Jacobian $A_t = \frac{\partial f}{\partial x}\big|_{x=\mu_{t-1}}$, then applies the standard Kalman update[@Thrun2005].

## Usage

```python
from faran.numpy import model

estimator = model.bicycle.estimator.ekf(
    time_step_size=0.1,
    wheelbase=2.5,
    rear_axle_distance=1.0,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

For unicycle models:

```python
estimator = model.unicycle.estimator.ekf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

## When to Use

Bicycle or unicycle obstacle models. Good default for nonlinear systems, but can diverge when the dynamics are highly nonlinear.

## API Reference

See the [model API reference](../../api/model.md) for full signatures.

\bibliography
