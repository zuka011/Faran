# Collectors

Collectors record simulation data for [evaluation metrics](../metrics/index.md), visualization, and debugging. They wrap planners and obstacle observers using a decorator pattern — the wrapped object behaves identically but also records data.

## How Collectors Work

```python
from faran import collectors, metrics

# Wrap the planner to collect state histories
planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states,
)

# Create a registry to manage collectors and metrics
registry = metrics.registry(
    collision_metric := metrics.collision(...),
    collectors=collectors.registry(planner),
)
```

## Available Collectors

| Collector | Wraps | Records |
|-----------|-------|---------|
| [States](states.md) | Planner | State at each planning step |
| [Obstacles](obstacles.md) | Observer | Obstacle states and forecasts |
| Risk | Planner | Risk metric values over time |

## Registry

The registry manages collectors and metrics. After the simulation loop, query it for metric results:

```python
from faran import access

collision_result = registry.get(collision_metric)
states = registry.data(access.states.require())
```

Use `.require()` to raise if data wasn't collected, or `.optional()` if the data might not be available.

## API Reference

See the [collectors API reference](../../api/collectors.md) for full signatures.
