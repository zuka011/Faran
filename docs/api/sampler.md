# sampler

Samplers generate control input perturbations around a nominal sequence for MPPI rollout exploration. For guidance on choosing and tuning samplers, see the [Samplers guide](../guide/samplers.md).

## Gaussian Sampler

Draws i.i.d. Gaussian perturbations per timestep with a fixed standard deviation per control dimension. Simple and fast — the default choice for most scenarios.

```python
from faran.numpy import sampler, types
import numpy as np

control_sampler = sampler.gaussian(
    standard_deviation=np.array([0.5, 0.2]),
    rollout_count=256,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

::: faran.samplers.gaussian.basic.NumPyGaussianSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.samplers.gaussian.accelerated.JaxGaussianSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Halton Spline Sampler

Generates temporally smooth perturbations using Halton quasi-random sequences interpolated through cubic splines. Provides better coverage of the sampling space (low discrepancy) and smoother control sequences than Gaussian sampling. See the [Samplers guide](../guide/samplers.md#choosing-a-sampler) for a comparison.

```python
control_sampler = sampler.halton(
    standard_deviation=np.array([0.5, 0.2]),
    rollout_count=256,
    knot_count=8,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

::: faran.samplers.halton.basic.NumPyHaltonSplineSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.samplers.halton.accelerated.JaxHaltonSplineSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Obstacle State Samplers

Obstacle state samplers draw from predicted obstacle state distributions (Gaussian) for risk-aware collision cost evaluation.

::: faran.obstacles.sampler.basic.NumPyGaussianObstacle2dPoseSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Sampler Protocol

::: faran.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3
