---
status: draft
---

# Waypoints Trajectory

Cubic-spline path through a sequence of 2D waypoints. Cumulative arc lengths are computed from the waypoints to build natural cubic spline interpolants for $x(\phi)$ and $y(\phi)$. Heading is derived from the spline tangent:

$$\theta(\phi) = \operatorname{atan2}\!\bigl(y'(\phi),\; x'(\phi)\bigr)$$

Closest-point queries use a coarse KD-tree (NumPy) or brute-force (JAX) lookup followed by Newton refinement.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | `(N, 2) array \| Sequence[tuple]` | — | Waypoint coordinates |
| `path_length` | `float \| None` | `None` | Parameterization length $L$. Defaults to natural arc length. |
| `coarse_search_samples` | `int` | `200` | Samples for the initial nearest-point lookup |
| `refining_iterations` | `int` | `3` | Newton iterations for closest-point refinement |

## Usage

```python
from faran.numpy import trajectory

reference = trajectory.waypoints(
    points=[(0.0, 0.0), (5.0, 3.0), (10.0, 0.0)],
    path_length=12.0,
)
```

## NumPy

::: faran.trajectories.waypoints.basic.NumPyWaypointsTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - query

## JAX

::: faran.trajectories.waypoints.accelerated.JaxWaypointsTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - query
