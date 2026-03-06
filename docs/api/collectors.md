# collectors

Collectors record simulation data (states, obstacles, risk values) during planning runs for later analysis, evaluation, and visualization. For usage patterns, see the [Data Collection guide](../guide/collectors.md).

## Overview

Collectors use a decorator pattern: wrap a planner or observer, and data is captured automatically without changing the planning logic. Register collectors with a `MetricRegistry` to compute [evaluation metrics](../guide/metrics.md).

## Usage

```python
from faran import collectors, metrics, access

# Wrap planner to collect states
planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states(
        physical=types.bicycle.state_sequence.of_states,
        virtual=types.simple.state_sequence.of_states,
    ),
)

# Register with metrics
registry = metrics.registry(
    collision_metric := metrics.collision(...),
    collectors=collectors.registry(planner),
)

# After simulation, access data
states = registry.data(access.states.require())
result = registry.get(collision_metric)
```

## CollectorRegistry

::: faran.collectors.registry.CollectorRegistry
    options:
      show_root_heading: true
      heading_level: 3

## Data Access

::: faran.collectors.access
    options:
      show_root_heading: true
      heading_level: 3

## Warning Types

::: faran.collectors.NoCollectedDataWarning
    options:
      show_root_heading: true
      heading_level: 3
