# Obstacle Collectors

Record observed and predicted obstacle data.

## Obstacle State Collector

Records observed obstacle states:

```python
from faran import collectors

observer = collectors.obstacle_states.decorating(observer)
```

## Obstacle Forecast Collector

Records predicted obstacle trajectories:

```python
observer = collectors.obstacle_forecasts.decorating(observer)
```

## API Reference

See the [collectors API reference](../../api/collectors.md) for full signatures.
