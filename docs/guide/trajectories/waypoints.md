# Waypoint Trajectories

Piecewise linear path through a sequence of 2D points.

## Usage

```python
from faran.numpy import trajectory
from numtypes import array

reference = trajectory.waypoints(
    points=array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [25.0, 15.0], [20.0, 25.0],
    ], shape=(5, 2)),
    path_length=50.0,
)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `points` | Array of 2D waypoints, shape `(N, 2)`. |
| `path_length` | Parameterization length. If equal to the geometric arc length, $\phi = 1$ corresponds to 1 meter. |

## API Reference

See the [trajectory API reference](../../api/trajectory.md) for full signatures.
