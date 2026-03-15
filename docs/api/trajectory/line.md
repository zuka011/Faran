---
status: draft
---

# Line Trajectory

Straight-line reference path between two endpoints. Heading is constant along the path:

$$\theta = \operatorname{atan2}(y_\text{end} - y_\text{start},\; x_\text{end} - x_\text{start})$$

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `tuple[float, float]` | — | Starting point $(x_0, y_0)$ |
| `end` | `tuple[float, float]` | — | Ending point $(x_1, y_1)$ |
| `path_length` | `float \| None` | `None` | Parameterization length $L$. Defaults to Euclidean distance between endpoints. |

## Usage

```python
from faran.numpy import trajectory

reference = trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0)
```

## NumPy

::: faran.trajectories.line.basic.NumPyLineTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - query

## JAX

::: faran.trajectories.line.accelerated.JaxLineTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - query
