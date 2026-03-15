---
status: draft
---

# Kalman Filter

Linear state estimator for the [integrator](../model/integrator.md) model[@Thrun2005]. Maintains a Gaussian belief $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ via prediction and update.

## Equations

**Predict:**

$$
\hat{\boldsymbol{\mu}}_t = A \, \boldsymbol{\mu}_{t-1}, \quad
\hat{\boldsymbol{\Sigma}}_t = A \, \boldsymbol{\Sigma}_{t-1} \, A^\top + R
$$

**Update:**

$$
K_t = \hat{\boldsymbol{\Sigma}}_t \, H^\top \bigl(H \, \hat{\boldsymbol{\Sigma}}_t \, H^\top + Q\bigr)^{-1}
$$

$$
\boldsymbol{\mu}_t = \hat{\boldsymbol{\mu}}_t + K_t \bigl(\mathbf{z}_t - H \, \hat{\boldsymbol{\mu}}_t\bigr), \quad
\boldsymbol{\Sigma}_t = (I - K_t H) \, \hat{\boldsymbol{\Sigma}}_t
$$

| Symbol | Description |
|--------|-------------|
| $A$ | State transition matrix (built from `time_step_size`) |
| $H$ | Observation matrix (maps full state to observed positions) |
| $R$ | Process noise covariance |
| $Q$ | Observation noise covariance |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | Integration step $\Delta t$ |
| `process_noise_covariance` | scalar / 1D / 2D | — | $R$ — model deviation |
| `observation_noise_covariance` | scalar / 1D / 2D | — | $Q$ — sensor noise |
| `initial_state_covariance` | 2D array \| `None` | `None` | $\Sigma_0$ — initial uncertainty |
| `observation_dimension` | `int \| None` | `None` | Required when covariances are scalar |
| `noise_model` | provider \| `None` | `None` | [Adaptive noise](noise.md) provider |

```python
from faran.numpy import model

estimator = model.integrator.estimator.kf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

::: faran.models.integrator.basic.NumPyKfIntegratorStateEstimator
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

::: faran.models.integrator.accelerated.JaxKfIntegratorStateEstimator
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

For nonlinear models (bicycle, unicycle), use the [EKF](ekf.md) or [UKF](ukf.md).

\bibliography
