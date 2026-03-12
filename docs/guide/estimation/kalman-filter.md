# Kalman Filter

For linear models (integrator). Maintains a Gaussian belief $(\mu, \Sigma)$ and updates it optimally using the Kalman gain[@Thrun2005].

## Equations

**Prediction:**

$$
\hat\mu_t = A \, \mu_{t-1}, \quad \hat\Sigma_t = A \, \Sigma_{t-1} \, A^\top + R
$$

**Update:**

$$
K_t = \hat\Sigma_t \, H^\top (H \, \hat\Sigma_t \, H^\top + Q)^{-1}
$$

$$
\mu_t = \hat\mu_t + K_t (z_t - H \, \hat\mu_t), \quad \Sigma_t = (I - K_t H) \, \hat\Sigma_t
$$

## Usage

```python
from faran.numpy import model

estimator = model.integrator.estimator.kf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

## When to Use

Integrator (constant-velocity) obstacle models. Optimal for linear-Gaussian systems.

## API Reference

See the [model API reference](../../api/model.md) for full signatures.

\bibliography
