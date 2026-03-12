# Cost Functions

Cost functions score each rollout at every time step. MPPI sums costs over time, then uses softmax weighting to combine rollouts — lower cost is better.

```python
costs_array = cost_function(inputs=control_batch, states=state_batch)
# costs_array.array has shape (T, M)
```

Three categories:

| Category | Purpose | Functions |
|----------|---------|-----------|
| [Tracking](tracking.md) | Follow a reference path | Contouring, lag, progress |
| [Safety](safety.md) | Avoid obstacles and boundaries | Collision, boundary |
| [Comfort](comfort.md) | Smooth control outputs | Control smoothing, control effort |

## Combining Costs

`costs.combined` sums any number of cost components:

```python
from faran.numpy import costs

total = costs.combined(
    costs.tracking.contouring(...),
    costs.tracking.lag(...),
    costs.tracking.progress(...),
    costs.safety.collision(...),
    costs.comfort.control_smoothing(...),
)
```

## Custom Cost Functions

Any callable with the right signature works:

```python
import numpy as np

def speed_limit_cost(*, inputs, states, target_speed=5.0, weight=10.0):
    speeds = states.array[:, 3, :]
    return types.numpy.costs(weight * (speeds - target_speed) ** 2)

total = costs.combined(
    costs.tracking.contouring(...),
    speed_limit_cost,
)
```

## API Reference

See the [costs API reference](../../api/costs.md) for full signatures.
