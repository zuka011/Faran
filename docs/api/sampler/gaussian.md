---
status: draft
---

# Gaussian Sampler

i.i.d. zero-mean Gaussian perturbations per timestep:

$$
\boldsymbol{\epsilon}_t^{(m)} \sim \mathcal{N}(\mathbf{0},\, \operatorname{diag}(\boldsymbol{\sigma}^2))
$$

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `standard_deviation` | array / sequence | — | Per-dimension $\boldsymbol{\sigma}$ |
| `rollout_count` | `int` | — | Number of samples $M$ |
| `to_batch` | batch creator | — | Wraps raw array into typed batch |
| `seed` | `int` | — | RNG seed (NumPy) |
| `key` | `PRNGKeyArray \| None` | `None` | JAX PRNG key (JAX only, alternative to `seed`) |

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
      heading_level: 2
      members:
        - create

::: faran.samplers.gaussian.accelerated.JaxGaussianSampler
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create
