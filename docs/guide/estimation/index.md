# State Estimation

State estimation recovers unobserved state variables (speed, acceleration, steering angle) and quantifies uncertainty from noisy position/heading observations. Estimators are primarily used for obstacle state estimation — tracking other vehicles and predicting their future motion.

## Why Estimate State?

A typical obstacle detector provides position $(x, y)$ and heading $\theta$. Prediction models need additional variables:

| Model | Additional variables needed |
|-------|---------------------------|
| [Bicycle](../models/bicycle.md) | Speed $v$, acceleration $a$, steering angle $\delta$ |
| [Unicycle](../models/unicycle.md) | Speed $v$, angular velocity $\omega$ |
| Integrator | Velocity components |

Estimation fills in these unobserved variables and provides a covariance matrix capturing how uncertain each state component is. This uncertainty propagates through motion prediction into [risk-aware collision costs](../costs/safety.md#collision).

## Available Estimators

| Estimator | Model Type | Covariance | Tuning | Cost |
|-----------|-----------|------------|--------|------|
| [Finite Difference](finite-difference.md) | Any | No | None | Cheapest |
| [Kalman Filter](kalman-filter.md) | Linear (integrator) | Yes | $R$, $Q$ | Low |
| [EKF](ekf.md) | Nonlinear | Yes | $R$, $Q$ | Medium |
| [UKF](ukf.md) | Nonlinear | Yes | $R$, $Q$, $\alpha$, $\beta$ | Highest |

If you just need obstacle positions without uncertainty, use Finite Difference. If you need covariance for risk-aware planning, use EKF or UKF.

## Noise Covariance

All Kalman-family estimators require two noise parameters:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| `process_noise_covariance` | $R$ | How much the true dynamics deviate from the model |
| `observation_noise_covariance` | $Q$ | How noisy the sensor observations are |

Both accept a scalar (expanded to diagonal), 1D array (diagonal entries), or 2D array (full matrix).

Larger process noise → estimator trusts observations more. Larger observation noise → estimator trusts the model more.

For [adaptive noise estimation](adaptive.md), the covariances are tuned online.

## API Reference

See the [model API reference](../../api/model.md) for estimator signatures and the [predictor API reference](../../api/predictor.md) for connecting estimators to motion prediction.
