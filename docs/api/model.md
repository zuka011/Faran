# model

Dynamical models define the state transition function $f(\mathbf{u}, \mathbf{x})$ used to simulate rollouts during MPPI planning. For a conceptual overview, see the [Dynamics Models guide](../guide/models.md).

## Kinematic Bicycle Model

The kinematic bicycle model [^1] represents a wheeled vehicle with four state variables and two control inputs, discretized via Euler integration. An optional rear axle distance $l_r$ shifts the reference point from the rear axle toward the center of gravity, introducing a slip angle $\beta$:

$$
\beta = \arctan\!\left(\frac{l_r}{L} \tan(\delta_t)\right)
$$

$$
\begin{gathered}
x_{t+1} = x_t + v_t \cos(\theta_t + \beta) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t + \beta) \, \Delta t \\
\theta_{t+1} = \theta_t + \frac{v_t}{L} \cos(\beta) \tan(\delta_t) \, \Delta t, \quad
v_{t+1} = v_t + a_t \, \Delta t
\end{gathered}
$$

where $L$ is the wheelbase, $l_r$ is the rear axle distance (default $0$), and $\Delta t$ the time step size. When $l_r = 0$, $\beta = 0$ and the equations reduce to the standard rear-axle model.

| Component | Variables |
|-----------|-----------|
| State | $[x, y, \theta, v]$ — position, heading, speed |
| Controls | $[a, \delta]$ — acceleration, steering angle |
| Parameters | $L$, $l_r$ — wheelbase, rear axle distance |

[^1]: P. Polack et al., "The Kinematic Bicycle Model: A Consistent Model for Planning Feasible Trajectories for Autonomous Vehicles?," IEEE IV, 2017.

::: faran.models.bicycle.basic.NumPyBicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.bicycle.accelerated.JaxBicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Unicycle Model

The unicycle model [^2] represents a point robot with direct velocity and angular velocity control:

$$
x_{t+1} = x_t + v_t \cos(\theta_t) \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t) \Delta t, \quad
\theta_{t+1} = \theta_t + \omega_t \Delta t
$$

| Component | Variables |
|-----------|-----------|
| State | $[x, y, \theta]$ — position, heading |
| Controls | $[v, \omega]$ — linear velocity, angular velocity |

[^2]: G. Oriolo, A. De Luca, M. Vendittelli, "WMR Control via Dynamic Feedback Linearization," IEEE TCST, 2002.

::: faran.models.unicycle.basic.NumPyUnicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.unicycle.accelerated.JaxUnicycleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Integrator Model

An $n$-dimensional integrator: $x_{t+1} = x_t + v_t \Delta t$. Used internally for the MPCC virtual state (path parameter $\phi$), but also available as a general-purpose model.

::: faran.models.integrator.basic.NumPyIntegratorModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.integrator.accelerated.JaxIntegratorModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## DynamicalModel Protocol

::: faran.types.DynamicalModel
    options:
      show_root_heading: true
      heading_level: 3

## Obstacle Models

Obstacle models are used by predictors to propagate obstacle states forward in time. They follow the same Euler integration as the corresponding dynamical models but operate on observed obstacle states.

::: faran.models.bicycle.basic.NumPyBicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.models.unicycle.basic.NumPyUnicycleObstacleModel
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## State Estimators

Estimators recover unobserved state variables and quantify uncertainty from noisy observations. For conceptual background, see [State Estimation](../guide/estimation.md).

### Bicycle Estimators

```python
from faran.numpy import model

# Finite difference (no covariance)
fd = model.bicycle.estimator.finite_difference(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
)

# Extended Kalman Filter
ekf = model.bicycle.estimator.ekf(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)

# Unscented Kalman Filter
ukf = model.bicycle.estimator.ukf(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

### Unicycle Estimators

```python
fd = model.unicycle.estimator.finite_difference(time_step_size=0.1)

ekf = model.unicycle.estimator.ekf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)

ukf = model.unicycle.estimator.ukf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

### Integrator Estimators

```python
fd = model.integrator.estimator.finite_difference(time_step_size=0.1)

kf = model.integrator.estimator.kf(
    time_step_size=0.1,
    process_noise_covariance=1e-3,
    observation_noise_covariance=1e-2,
)
```

## Noise Models

Noise models adapt filter covariances at runtime. See [State Estimation](../guide/estimation.md#adaptive-noise) for usage.

```python
from faran.numpy import noise

# Fixed noise (default)
identity = noise.identity

# Adaptive (IAE method)
adaptive = noise.adaptive(window_size=10)

# Clamped (floor on diagonal entries)
clamped = noise.clamped(
    noise.adaptive(window_size=10),
    floor=noise.covariances(
        process=1e-5,
        observation=1e-5,
        process_dimension=6,
        observation_dimension=3,
    ),
)
```
