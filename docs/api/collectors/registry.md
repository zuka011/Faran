---
status: draft
---

# Collector Components

## CollectorRegistry

Aggregates multiple collectors and produces unified `SimulationData`. Automatically invalidates cached data when any collector receives new entries.

```python
from faran import collectors

registry = collectors.registry(state_collector, obstacle_collector)
data = registry.data  # SimulationData dict
```

::: faran.collectors.registry.CollectorRegistry
    options:
      show_root_heading: true
      heading_level: 3

## Collector Protocol

::: faran.collectors.registry.Collector
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
