# Dynamics Models

Dynamics models define how state evolves given a control input. They are the prediction engine in MPPI: at each planning step, the planner simulates $M$ rollouts through the model to evaluate different control sequences.

Three models are available. All support both NumPy and JAX backends, with configurable state and input limits.

| Model | State | Controls | Parameters |
|-------|-------|----------|------------|
| [Kinematic Bicycle](bicycle.md) | $x, y, \theta, v$ | $a, \delta$ | Wheelbase, rear axle distance |
| [Unicycle](unicycle.md) | $x, y, \theta$ | $v, \omega$ | — |
| Integrator | $x_1, \ldots, x_n$ | $u_1, \ldots, u_n$ | — |

## Integrator

A generic $n$-dimensional single integrator: $x_{t+1} = x_t + u_t \, \Delta t$. Used internally for the [MPCC](../concepts/mpcc.md) virtual state (path parameter $\phi$).

```python
from faran.numpy import model

integrator = model.integrator.dynamical(time_step_size=0.1)
```

## Obstacle Models

Each dynamics model has a corresponding obstacle variant for predicting how obstacles move. These follow the same kinematics but operate on observed states and are paired with [state estimation](../estimation/index.md).

```python
obstacle_model = model.bicycle.obstacle(
    time_step_size=0.1, wheelbase=2.5, rear_axle_distance=1.0,
)
```

## State Limits

All models clamp values after each integration step when limits are specified:

```python
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

Omitting a limit leaves that dimension unconstrained.

## API Reference

See the [model API reference](../../api/model.md) for full signatures.
