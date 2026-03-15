---
status: draft
---

# Cost Functions

Cost functions score each rollout at every time step. MPPI sums costs over the horizon and applies softmax weighting — lower total cost is better.

| Category | Functions | Purpose |
|----------|-----------|---------|
| [Tracking](tracking.md) | Contouring, Lag, Progress | Follow a reference path (MPCC) |
| [Safety](safety.md) | Collision, Boundary | Obstacle avoidance, corridor constraints |
| [Comfort](comfort.md) | Control Smoothing, Control Effort | Smooth actuator commands |

## CostFunction Protocol

::: faran.types.CostFunction
    options:
      show_root_heading: true
      heading_level: 3

## Combining Costs

`costs.combined` sums any number of cost components elementwise:

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

::: faran.costs.combined.CombinedCost
    options:
      show_root_heading: true
      heading_level: 3

Any callable matching the `CostFunction` protocol can be passed to `costs.combined`.
