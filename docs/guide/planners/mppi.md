# MPPI

Model Predictive Path Integral (MPPI) is a sampling-based trajectory optimizer[@Williams2015][@Williams2016][@Williams2017]. At each step it:

1. Draws $M$ control perturbations around a nominal sequence
2. Simulates each through a dynamics model to produce $M$ rollouts
3. Evaluates a cost function per rollout
4. Computes a softmax-weighted average as the optimal control

$$
u^* = \sum_{m=1}^{M} w_m \, u_m, \quad w_m = \frac{1}{\eta} \exp\!\left(-\frac{1}{\lambda}\,(J_m - J_{\min})\right)
$$

## Configuration

```python
from faran.numpy import mppi, model, sampler, costs, types
import numpy as np

planner = mppi.base(
    model=model.bicycle.dynamical(time_step_size=0.1, wheelbase=2.5),
    cost_function=costs.combined(...),
    sampler=sampler.gaussian(
        standard_deviation=np.array([0.5, 0.05]),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
)
```

## Temperature

The temperature $\lambda$ controls the sharpness of the softmax weighting.

- **Low** ($\lambda \approx 1$–$10$) — concentrates weight on the lowest-cost samples. More greedy.
- **High** ($\lambda \approx 50$–$100$) — distributes weight more evenly. More exploration, smoother control.

Set $\lambda$ so that $(J_m - J_{\min}) / \lambda$ falls roughly in $[0.1, 10]$ for most rollouts. If all weights collapse to zero, increase $\lambda$. If all weights are nearly uniform, decrease it.

## Sampler Seeding

Samplers are deterministic given a seed. When using `mppi.augmented`, use different seeds for physical and virtual samplers:

```python
physical_sampler = sampler.gaussian(seed=42, ...)
virtual_sampler = sampler.gaussian(seed=43, ...)
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

## API Reference

See the [MPPI API reference](../../api/mppi.md) for full signatures.

\bibliography
