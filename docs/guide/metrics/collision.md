# Collision Metric

Detects collisions and reports minimum distances over the simulation.

## Usage

```python
from faran import metrics

collision_metric = metrics.collision(
    distance_threshold=0.5,
    distance=distance_extractor,
)
```

## Results

```python
result = registry.get(collision_metric)
result.distances           # (T, V) — distance per time step per vehicle part
result.min_distances       # (V,) — minimum over time
result.collisions          # (T, V) — boolean collision flags
result.collision_detected  # bool — any collision at any time
```

## API Reference

See the [metrics API reference](../../api/metrics.md) for full signatures.
