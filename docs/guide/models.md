# Dynamics Models

Dynamics models define how state evolves given a control input. They are the core prediction engine in MPPI: at each planning step, the planner simulates $M$ rollouts through the model to evaluate different control sequences.

Faran provides three models. All support both NumPy and JAX backends, and each has configurable state/input limits.

## Kinematic Bicycle Model

The standard model for wheeled vehicles[@Polack2017]. Four state variables and two control inputs, discretized via Euler integration. The model supports an optional **rear axle distance** $l_r$ that shifts the reference point from the rear axle toward the center of gravity:

$$
\beta = \arctan\!\left(\frac{l_r}{L} \tan(\delta_t)\right)
$$

$$
x_{t+1} = x_t + v_t \cos(\theta_t + \beta) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t + \beta) \, \Delta t
$$

$$
\theta_{t+1} = \theta_t + \frac{v_t}{L} \cos(\beta) \tan(\delta_t) \, \Delta t, \quad
v_{t+1} = v_t + a_t \, \Delta t
$$

where $L$ is the wheelbase, $l_r$ is the distance from the rear axle to the reference point, and $\beta$ is the slip angle. When $l_r = 0$ (the default), the reference point is at the rear axle and the equations simplify to the standard form ($\beta = 0$).

| Component | Variables | Meaning |
|-----------|-----------|---------|
| State | $[x, y, \theta, v]$ | Position, heading, speed |
| Controls | $[a, \delta]$ | Acceleration, steering angle |
| Parameters | $L$, $l_r$ | Wheelbase, rear axle distance |

```python
from faran.numpy import model

bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

To place the reference point at the center of gravity, set `rear_axle_distance` to the distance from the rear axle:

```python
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    rear_axle_distance=1.0,  # reference point 1m ahead of rear axle
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

**When to use:** Vehicles with front-axle steering (cars, trucks). The bicycle approximation is valid at moderate speeds and steering angles where tire slip is negligible. Use `rear_axle_distance` when the tracked position is not the rear axle (e.g., a GPS antenna or center of gravity).

## Unicycle Model

A simpler model for differential-drive or omnidirectional robots[@Oriolo2002]. Three state variables with direct velocity control:

$$
x_{t+1} = x_t + v_t \cos(\theta_t) \, \Delta t, \quad
y_{t+1} = y_t + v_t \sin(\theta_t) \, \Delta t, \quad
\theta_{t+1} = \theta_t + \omega_t \, \Delta t
$$

| Component | Variables | Meaning |
|-----------|-----------|---------|
| State | $[x, y, \theta]$ | Position, heading |
| Controls | $[v, \omega]$ | Linear velocity, angular velocity |

```python
unicycle = model.unicycle.dynamical(
    time_step_size=0.1,
    speed_limits=(0.0, 5.0),
    angular_velocity_limits=(-1.0, 1.0),
)
```

**When to use:** Robots or agents where you control velocity and turning rate directly, or when you don't need a detailed vehicle model.

## Integrator Model

A generic $n$-dimensional single integrator: $x_{t+1} = x_t + u_t \, \Delta t$. Used internally for the MPCC virtual state (path parameter $\phi$), but also available as a general-purpose model.

```python
integrator = model.integrator.dynamical(time_step_size=0.1)
```

**When to use:** Virtual states in MPCC (handled automatically by `mppi.mpcc`), or any scenario where you need a simple integrator as a building block.

## Obstacle Models

Each dynamics model has a corresponding **obstacle model** for predicting how obstacles move. Obstacle models follow the same kinematics but operate on observed states and are paired with state estimation (see [State Estimation](estimation.md)).

```python
obstacle_model = model.bicycle.obstacle(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
)
```

Obstacle models are used by [predictors](obstacles.md#obstacle-state-provider) to propagate obstacle states forward over the planning horizon.

## State Limits

All models support optional state and input limits. When specified, the model clamps values after each integration step:

```python
# Speed clamped to [0, 15], steering to [-0.5, 0.5], acceleration to [-3, 3]
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

Omitting a limit (or passing `None`) leaves that dimension unconstrained.

## API Reference

See the [model API reference](../api/model.md) for full signatures and docstrings.

\bibliography
