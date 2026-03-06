# Data Collection

Collectors record simulation data for [evaluation metrics](metrics.md), visualization, and debugging. They wrap planners and obstacle observers, intercepting calls to store state histories without changing the planning logic.

## How Collectors Work

Collectors use a **decorator pattern**: you wrap a planner (or observer) with a collector, and the wrapped object behaves identically but also records data. The collected data is accessed through a **registry**.

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

### State Collector

Records the state at each planning step:

```python
planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states,
)
```

The `transformer` converts individual states into a state sequence. Use the appropriate type for your model:

| Model | Transformer |
|-------|-------------|
| Augmented (MPCC) | `types.augmented.state_sequence.of_states` |
| Bicycle | `types.bicycle.state_sequence.of_states` |
| Unicycle | `types.unicycle.state_sequence.of_states` |
| Simple | `types.simple.state_sequence.of_states` |

For augmented states, you need to specify both physical and virtual transformers:

```python
planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states(
        physical=types.bicycle.state_sequence.of_states,
        virtual=types.simple.state_sequence.of_states,
    ),
)
```

### Obstacle State Collector

Records observed obstacle states:

```python
observer = collectors.obstacle_states.decorating(observer)
```

### Obstacle Forecast Collector

Records predicted obstacle trajectories:

```python
observer = collectors.obstacle_forecasts.decorating(observer)
```

### Risk Collector

Records risk metric values over time:

```python
planner = collectors.risk.decorating(planner)
```

## Registry

The registry manages the relationship between collectors and metrics. After the simulation loop, query it for metric results:

```python
from faran import access

# Get metric results
collision_result = registry.get(collision_metric)

# Access raw collected data
states = registry.data(access.states.require())
```

## Accessing Data

Use the `access` namespace for type-safe data retrieval:

```python
from faran import access

states = registry.data(access.states.require())
obstacle_states = registry.data(access.obstacle_states.require())
```

The `.require()` variant raises an error if the data hasn't been collected. Use `.optional()` if the data might not be available.

## API Reference

See the [collectors API reference](../api/collectors.md) for full signatures.
