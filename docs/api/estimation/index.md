---
status: draft
---

# State Estimation

Recovers unobserved state variables and quantifies uncertainty from noisy pose observations. For conceptual background, see the [obstacle avoidance guide](../../guide/examples/obstacle-avoidance.md).

## Observed vs. Unobserved Variables

| Model | Observed | Unobserved |
|-------|----------|------------|
| [Bicycle](../model/bicycle.md) | $x, y, \theta$ | $v, a, \delta$ |
| [Unicycle](../model/unicycle.md) | $x, y, \theta$ | $v, \omega$ |
| [Integrator](../model/integrator.md) | $\mathbf{x}$ | $\mathbf{v}$ |

## Estimator Summary

| Estimator | Applicable Models | Covariance | Parameters |
|-----------|-------------------|------------|------------|
| [Finite Difference](finite-difference.md) | All | No | `time_step_size` |
| [Kalman Filter](kalman.md) | Integrator (linear) | Yes | $R$, $Q$ |
| [EKF](ekf.md) | Bicycle, Unicycle | Yes | $R$, $Q$ |
| [UKF](ukf.md) | Bicycle, Unicycle | Yes | $R$, $Q$, $\alpha$, $\beta$ |

All Kalman-family estimators accept an optional [noise model](noise.md) for adaptive covariance tuning.
