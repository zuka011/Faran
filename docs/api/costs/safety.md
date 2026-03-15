---
status: draft
---

# Safety Costs

## Collision Cost

Hinge-loss collision avoidance[@Schulman2013]. Activates when any vehicle part's signed distance falls below its threshold:

$$
J_{\text{col}} = \sum_{i=1}^{V} k_{\text{col}} \max(0,\; d_0^{(i)} - d_i)
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `obstacle_states` | state provider | Predicted obstacle states |
| `sampler` | obstacle sampler | Draws position samples from belief |
| `distance` | distance extractor | Computes signed distances |
| `distance_threshold` | array of shape $(V,)$ | Per-part threshold $d_0^{(i)}$ |
| `weight` | `float` | $k_{\text{col}}$ |
| `metric` | risk metric \| `None` | [Risk metric](risk.md) for uncertainty-aware evaluation |

```python
from faran.numpy import costs, obstacles, risk
import numpy as np

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=np.array([0.5, 0.5, 0.5]),
    weight=1500.0,
    metric=risk.cvar(alpha=0.95, sample_count=50),
)
```

::: faran.costs.collision.basic.NumPyCollisionCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.collision.accelerated.JaxCollisionCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Boundary Cost

Same hinge-loss formulation applied to corridor boundary distances. See [Boundary](../boundary/index.md) for available boundary types.

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance` | boundary distance extractor | Computes lateral boundary distance |
| `distance_threshold` | `float` | Activation threshold $d_0$ |
| `weight` | `float` | $k_{\text{boundary}}$ |

::: faran.costs.boundary.basic.NumPyBoundaryCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.boundary.accelerated.JaxBoundaryCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Distance Functions

### Circle Distance

Pairwise signed distances between circular approximations of ego and obstacle[@Tolksdorf2024].

::: faran.costs.distance.circles.basic.NumPyCircleDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

::: faran.costs.distance.circles.accelerated.JaxCircleDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

### SAT Distance

Signed distances between convex polygons via the Separating Axis Theorem[@Gottschalk1997].

::: faran.costs.distance.sat.basic.NumPySatDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

::: faran.costs.distance.sat.accelerated.JaxSatDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

\bibliography
