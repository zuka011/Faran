---
status: draft
---

# Noise Models

All Kalman-family estimators ([KF](kalman.md), [EKF](ekf.md), [UKF](ukf.md)) require process and observation noise covariances.

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `process_noise_covariance` | $R$ | Model deviation from true dynamics |
| `observation_noise_covariance` | $Q$ | Sensor noise |

Both accept a scalar (isotropic), 1D array (diagonal), or 2D array (full matrix).

## Covariance Helper

Standardizes scalar/vector/matrix noise descriptions into full covariance matrices.

```python
from faran.numpy import noise

covariances = noise.covariances(
    process=1e-3,
    observation=1e-2,
    process_dimension=6,
    observation_dimension=3,
)
```

## Adaptive Noise

Innovation-based Adaptive Estimation (IAE)[@Mohamed1999] adapts covariances online from the innovation sequence over a sliding window.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | — | Past observations used for adaptation |

```python
from faran.numpy import noise

adaptive = noise.adaptive(window_size=10)
```

::: faran.filters.noise.basic.NumPyAdaptiveNoiseProvider
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.filters.noise.accelerated.JaxAdaptiveNoiseProvider
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Clamped Noise

Decorator that clamps eigenvalues of an inner noise model's output to a floor and/or ceiling, preventing overconfidence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inner` | noise provider | — | Provider to wrap |
| `floor` | `NoiseCovarianceBounds \| None` | `None` | Minimum eigenvalue bounds |
| `ceiling` | `NoiseCovarianceBounds \| None` | `None` | Maximum eigenvalue bounds |

```python
clamped = noise.clamped(
    noise.adaptive(window_size=10),
    floor=noise.covariance_bounds(process=1e-5, observation=1e-5),
)
```

::: faran.filters.noise.basic.NumPyClampedNoiseProvider
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - decorate

::: faran.filters.noise.accelerated.JaxClampedNoiseProvider
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - decorate

## Identity Noise

Default no-op provider. Returns the input covariances unchanged. Used when no `noise_model` is specified.

::: faran.filters.noise.common.IdentityNoiseModelProvider
    options:
      show_root_heading: true
      heading_level: 3

\bibliography
