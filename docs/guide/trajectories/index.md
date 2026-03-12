# Trajectories

Trajectories define reference paths for path-following formulations like [MPCC](../concepts/mpcc.md). Two types are available: [waypoints](waypoints.md) and [line](line.md).

## Querying

Query a trajectory at specific path parameters to get reference points:

```python
from faran.numpy import types
from numtypes import array

path_params = types.path_parameters(
    array([[0.0, 5.0], [2.5, 7.5]], shape=(2, 2))
)

ref_points = reference.query(path_params)
ref_points.x()        # shape: (T, M)
ref_points.y()        # shape: (T, M)
ref_points.heading()  # shape: (T, M)
```

## Properties

```python
reference.path_length     # user-specified parameterization length
reference.natural_length  # actual geometric arc length
reference.end             # (x, y) of the final point
```

## Cyclic Trajectories

For looped paths (e.g., race tracks), enable periodicity in the MPCC config:

```python
planner, model, _, _ = mppi.mpcc(
    ...,
    reference=reference,
    config={"virtual": {"periodic": True}},
)
```

## API Reference

See the [trajectory API reference](../../api/trajectory.md) for full signatures.
