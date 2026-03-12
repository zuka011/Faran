# Variable-Width Corridor

Width changes at arc-length breakpoints along the reference path.

## Usage

```python
from faran.numpy import boundary

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

## Parameters

| Parameter | Description |
|-----------|-------------|
| `reference` | Reference trajectory defining the corridor center. |
| `position_extractor` | Extracts positions from state batches. |
| `widths` | Dict mapping arc-length positions to `{"left": ..., "right": ...}` widths. |

Widths are constant between breakpoints. Breakpoints must be non-decreasing arc-length values.

## API Reference

See the [boundary API reference](../../api/boundary.md) for full signatures.
