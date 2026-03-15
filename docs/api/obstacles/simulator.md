---
status: draft
---

# Obstacle Simulators

Ground-truth obstacle generators for testing and simulation. In production, observations come from a perception system.

## Static Obstacle Simulator

Obstacles with fixed positions that do not move.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positions` | `(K, 2) array` | — | Obstacle positions |
| `headings` | `(K,) array \| None` | `None` | Obstacle headings. Defaults to zero. |

### Usage

```python
from faran.numpy import obstacles
from numtypes import array

simulator = obstacles.static(
    positions=array([[3.0, 1.0], [7.0, -1.0]], shape=(2, 2)),
)
```

::: faran.obstacles.static.basic.NumPyStaticObstacleSimulator
    options:
      show_root_heading: true
      heading_level: 3

::: faran.obstacles.static.accelerated.JaxStaticObstacleSimulator
    options:
      show_root_heading: true
      heading_level: 3

## Dynamic Obstacle Simulator

Obstacles moving with constant velocity. Heading is derived from velocity direction.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `positions` | `(K, 2) array` | Initial obstacle positions |
| `velocities` | `(K, 2) array` | Constant velocity vectors $(v_x, v_y)$ per obstacle |

### Usage

```python
simulator = obstacles.dynamic(
    positions=array([[5.0, 2.0]], shape=(1, 2)),
    velocities=array([[-1.0, 0.0]], shape=(1, 2)),
)
```

::: faran.obstacles.dynamic.basic.NumPyDynamicObstacleSimulator
    options:
      show_root_heading: true
      heading_level: 3

::: faran.obstacles.dynamic.accelerated.JaxDynamicObstacleSimulator
    options:
      show_root_heading: true
      heading_level: 3
