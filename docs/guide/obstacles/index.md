# Obstacles

The obstacle avoidance pipeline has three stages:

```
Obstacle states → Distance computation → Collision cost
```

When obstacle positions are uncertain, a [risk metric](../risk.md) replaces the deterministic collision cost with a risk-sensitive evaluation.

| Component | Page | Description |
|-----------|------|-------------|
| [Distance](distance.md) | Distance computation | Circle-to-circle or SAT signed distances |
| [Prediction](prediction.md) | Motion prediction | Curvilinear forward propagation of obstacle states |
| [ID Assignment](id-assignment.md) | Frame-to-frame tracking | Hungarian algorithm for persistent obstacle IDs |

## Collision Cost

The collision cost penalizes rollouts where the signed distance drops below a threshold:

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

For risk-aware evaluation, pass a [risk metric](../risk.md):

```python
from faran.numpy import risk

collision = costs.safety.collision(
    ...,
    metric=risk.cvar(alpha=0.95, sample_count=50),
)
```

## API Reference

See the [obstacles API reference](../../api/obstacles.md) for full signatures.
