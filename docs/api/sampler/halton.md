---
status: draft
---

# Halton Spline Sampler

Quasi-random perturbations with temporal smoothness[@Halton1960]. Halton sequences in $D_u \times K$ dimensions produce low-discrepancy uniform samples, transformed to Gaussian via the inverse normal CDF, then interpolated across the time horizon with natural cubic splines.

| Parameter            | Type             | Default | Description                                |
|----------------------|------------------|---------|--------------------------------------------|
| `standard_deviation` | array / sequence | —       | Per-dimension $\boldsymbol{\sigma}$        |
| `rollout_count`      | `int`            | —       | Number of samples $M$                      |
| `knot_count`         | `int`            | —       | Spline knot count $K$ ($\geq 2$, $\leq T$) |
| `to_batch`           | batch creator    | —       | Wraps raw array into typed batch           |
| `seed`               | `int`            | —       | Halton scramble seed / start index         |

Maximum Halton dimension: $D_u \times K \leq 180$ (JAX backend).

```python
from faran.numpy import sampler, types
import numpy as np

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
      heading_level: 2
      members:
        - create

::: faran.samplers.halton.accelerated.JaxHaltonSplineSampler
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

\bibliography
