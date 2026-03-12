# Planners

A planner generates optimal control sequences by sampling, simulating, and scoring many candidate trajectories. Faran currently provides one planner: [MPPI](mppi.md).

## Factory Functions

| Factory          | Use case                                                       |
|------------------|----------------------------------------------------------------|
| `mppi.base`      | Custom MPC with your own model and cost function               |
| `mppi.augmented` | States with multiple components (e.g., physical + virtual)     |
| `mppi.mpcc`      | [MPCC](../concepts/mpcc.md) path following — highest-level API |

```python
from faran.numpy import mppi

# Bring your own model, cost, sampler
planner = mppi.base(model=..., cost_function=..., sampler=...)

# MPCC path following in one call
planner, model, contouring, lag = mppi.mpcc(model=..., sampler=..., reference=..., ...)
```

## Planning Loop

```python
state = initial_state
nominal = initial_nominal_input

for step in range(max_steps):
    control = planner.step(
        temperature=50.0, nominal_input=nominal, initial_state=state,
    )
    state = model.step(inputs=control.optimal, state=state)
    nominal = control.nominal
```

`control.optimal` is the weighted-average control sequence. `control.nominal` is the shifted sequence used as the sampling center for the next step.

## API Reference

See the [MPPI API reference](../../api/mppi.md) for full signatures.
