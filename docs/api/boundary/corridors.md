---
status: draft
---

# Boundary Corridors

## Fixed-Width Boundary

Constant corridor width on each side of the reference path. The signed distance is:

$$d = \min\bigl(w_\text{left} + e_\text{lat},\; w_\text{right} - e_\text{lat}\bigr)$$

where $e_\text{lat}$ is the lateral offset from the reference and $w_\text{left}$, $w_\text{right}$ are the left and right widths. Positive $d$ means the vehicle is inside the corridor.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | `Trajectory` | Reference trajectory defining the corridor center |
| `position_extractor` | `PositionExtractor` | Extracts $(x, y)$ positions from states |
| `left` | `float` | Left-side corridor width |
| `right` | `float` | Right-side corridor width |

### Usage

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

### NumPy

::: faran.costs.boundary.basic.NumPyFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

### JAX

::: faran.costs.boundary.accelerated.JaxFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

## Piecewise Fixed-Width Boundary

Width changes at arc-length breakpoints along the reference path. Between breakpoints, widths are constant. The distance formula is the same as fixed-width but uses segment-specific widths looked up by longitudinal position.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | `Trajectory` | Reference trajectory defining the corridor center |
| `position_extractor` | `PositionExtractor` | Extracts $(x, y)$ positions from states |
| `widths` | `dict[float, BoundaryWidths]` | Mapping from arc-length breakpoints to `{"left": float, "right": float}` |

### Usage

```python
corridor = boundary.piecewise_fixed_width(
    reference=reference,
    position_extractor=position_extractor,
    widths={
        0.0: {"left": 2.0, "right": 4.0},
        5.0: {"left": 3.0, "right": 5.0},
        7.0: {"left": 1.0, "right": 2.0},
    },
)
```

### NumPy

::: faran.costs.boundary.basic.NumPyPiecewiseFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

### JAX

::: faran.costs.boundary.accelerated.JaxPiecewiseFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create
