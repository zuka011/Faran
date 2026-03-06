# Samplers

Samplers generate control input perturbations around a nominal sequence. At each MPPI step, the sampler produces $M$ perturbed control sequences that the planner evaluates via rollout simulation. The quality and diversity of samples directly affects planning performance.

## Gaussian Sampler

Draws independent Gaussian perturbations at each time step with a fixed standard deviation per control dimension.

```python
from faran.numpy import sampler, types
import numpy as np

control_sampler = sampler.gaussian(
    standard_deviation=np.array([0.5, 0.05]),
    rollout_count=256,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

| Parameter            | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| `standard_deviation` | Per-dimension noise scale. Shape `(D_u,)`.                               |
| `rollout_count`      | Number of sample trajectories ($M$).                                     |
| `to_batch`           | Factory function that constructs a `ControlInputBatch` from a raw array. |
| `seed`               | Random seed for reproducibility.                                         |

**Characteristics:** Simple, fast, easy to tune. Each time step is sampled independently, which can produce jerky control sequences. Good default choice for prototyping.

## Halton Spline Sampler

Uses Halton quasi-random sequences mapped through an inverse normal CDF and interpolated with cubic splines. The result is temporally smooth, low-discrepancy perturbations.

```python
control_sampler = sampler.halton(
    standard_deviation=np.array([0.5, 0.05]),
    rollout_count=256,
    knot_count=8,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

| Parameter    | Description                                                                    |
|--------------|--------------------------------------------------------------------------------|
| `knot_count` | Number of spline knot points. More knots = more flexibility; fewer = smoother. |

All other parameters are the same as the Gaussian sampler.

**Characteristics:** Produces smoother control trajectories than Gaussian sampling, which can improve planner convergence and reduce control jitter. Better coverage of the sampling space (low discrepancy). Slightly more expensive per sample.

## Choosing a Sampler

| Criterion      | Gaussian                          | Halton Spline                     |
|----------------|-----------------------------------|-----------------------------------|
| **Smoothness** | Independent per time step         | Temporally correlated via splines |
| **Coverage**   | Pseudo-random (can cluster)       | Quasi-random (more uniform)       |
| **Cost**       | Cheapest                          | Slightly more expensive           |
| **Tuning**     | Standard deviation only           | Standard deviation + knot count   |
| **Best for**   | Quick prototyping, short horizons | Long horizons, smooth dynamics    |

Start with Gaussian. Switch to Halton if you see noisy control outputs or want better exploration with fewer rollouts.

## Standard Deviation Tuning

The standard deviation controls the exploration radius around the nominal control sequence. Too small and the planner gets stuck in local optima; too large and most samples are infeasible.

For a bicycle model with acceleration and steering controls, typical starting values are:

- **Acceleration:** 0.3–1.0 (depends on acceleration limits)
- **Steering:** 0.02–0.1 (smaller because steering has tighter limits)

A good rule of thumb: set the standard deviation to about 10–30% of the control limit range.

## API Reference

See the [sampler API reference](../api/sampler.md) for full signatures and docstrings.
