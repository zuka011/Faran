# Gaussian Sampler

Draws independent Gaussian perturbations at each time step with a fixed standard deviation per control dimension.

## Usage

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

| Parameter | Description |
|-----------|-------------|
| `standard_deviation` | Per-dimension noise scale. Shape `(D_u,)`. |
| `rollout_count` | Number of sample trajectories ($M$). |
| `to_batch` | Factory that constructs a `ControlInputBatch` from a raw array. |
| `seed` | Random seed for reproducibility. |

## Characteristics

Each time step is sampled independently, which can produce jerky control sequences. Simple, fast, easy to tune. Good default for prototyping.

## API Reference

See the [sampler API reference](../../api/sampler.md) for full signatures.
