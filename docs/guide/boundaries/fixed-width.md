# Fixed-Width Corridor

Constant width on each side of the reference path.

## Usage

```python
from faran.numpy import boundary, trajectory, types

reference = trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0)

corridor = boundary.fixed_width(
    reference=reference,
    position_extractor=lambda states: types.positions(
        x=states.array[:, 0, :], y=states.array[:, 1, :],
    ),
    left=2.0,
    right=5.0,
)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `reference` | Reference trajectory defining the corridor center. |
| `position_extractor` | Extracts positions from state batches. |
| `left` | Width to the left of the path (meters). |
| `right` | Width to the right of the path (meters). |

## Boundary Cost

```python
from faran.numpy import costs

boundary_cost = costs.safety.boundary(
    distance=corridor,
    distance_threshold=0.25,
    weight=1000.0,
)
```

The cost activates when the signed distance drops below `distance_threshold`.

## API Reference

See the [boundary API reference](../../api/boundary.md) for full signatures.
