# Safety Costs

Safety costs penalize proximity to obstacles and corridor boundaries.

## Collision

Penalizes rollouts where the signed distance to an obstacle drops below a threshold[@Schulman2013]:

$$
J_{\text{col}} = k_{\text{col}} \max(d_0 - d, \; 0)
$$

```python
from faran.numpy import costs, obstacles
from numtypes import array

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
)
```

The `distance_threshold` array has one entry per ego part (e.g., one per circle).

For risk-aware evaluation with uncertain obstacle positions, pass a risk metric:

```python
from faran.numpy import risk

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
    metric=risk.cvar(alpha=0.95, sample_count=50),
)
```

See [Obstacles](../obstacles/index.md) for distance computation and [Risk Metrics](../risk.md) for available metrics.

## Boundary

Penalizes states approaching the edges of a corridor:

```python
boundary_cost = costs.safety.boundary(
    distance=corridor,
    distance_threshold=0.25,
    weight=1000.0,
)
```

See [Boundaries](../boundaries/index.md) for corridor setup.

## API Reference

See the [costs API reference](../../api/costs.md) for full signatures.

\bibliography
