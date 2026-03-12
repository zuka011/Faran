# Halton Spline Sampler

Uses Halton quasi-random sequences mapped through an inverse normal CDF and interpolated with cubic splines. Produces temporally smooth, low-discrepancy perturbations.

## Usage

```python
from faran.numpy import sampler, types
import numpy as np

control_sampler = sampler.halton(
    standard_deviation=np.array([0.5, 0.05]),
    rollout_count=256,
    knot_count=8,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

| Parameter | Description |
|-----------|-------------|
| `knot_count` | Number of spline knot points. More knots = more flexibility; fewer = smoother. |

All other parameters are the same as the [Gaussian sampler](gaussian.md).

## Characteristics

Smoother control trajectories than Gaussian sampling — can improve planner convergence and reduce control jitter. Better coverage of the sampling space (low discrepancy). Slightly more expensive per sample.

Best for long horizons and smooth dynamics.

## API Reference

See the [sampler API reference](../../api/sampler.md) for full signatures.
