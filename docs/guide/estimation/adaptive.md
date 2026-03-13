# Adaptive Noise Estimation

When the true noise covariances are unknown or change over time, **Innovation-based Adaptive Estimation (IAE)**[@Mohamed1999] adapts them online using the observation innovation sequence.

The adaptive noise model monitors the innovation over a sliding window and adjusts both process and observation noise covariances to match the observed statistics.

## Usage

```python
from faran.numpy import model, noise

adaptive = noise.adaptive(window_size=10)

estimator = model.bicycle.estimator.ekf(
    time_step_size=0.1,
    wheelbase=2.5,
    rear_axle_distance=1.0,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
    noise_model=adaptive,
)
```

## Clamped Noise

The adaptive model may produce very small noise values, causing the filter to become overconfident. The **clamped** decorator enforces a minimum floor on eigenvalues:

```python
clamped = noise.clamped(
    noise.adaptive(window_size=10),
    floor=noise.covariance_bounds(
        process=1e-5,
        observation=1e-5,
    ),
)
```

This is composable — clamp any noise model, not just the adaptive one.

## API Reference

See the [model API reference](../../api/model.md) for full signatures.

\bibliography
