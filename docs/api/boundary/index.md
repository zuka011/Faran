---
status: draft
---

# boundary

Drivable corridor constraints around a reference [trajectory](../trajectory/index.md). Boundaries compute signed distances that feed into a [boundary cost](../costs/safety.md#boundary-cost).

## Distance Convention

| Value | Meaning |
|-------|---------|
| $d > 0$ | Inside corridor |
| $d = 0$ | On boundary |
| $d < 0$ | Violation |

## Available Corridors

| Corridor | Description | Factory |
|----------|-------------|---------|
| [Fixed-Width](corridors.md#fixed-width-boundary) | Constant left/right widths | `boundary.fixed_width(...)` |
| [Piecewise Fixed-Width](corridors.md#piecewise-fixed-width-boundary) | Widths varying at arc-length breakpoints | `boundary.piecewise_fixed_width(...)` |

## Protocols

::: faran.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - left
        - right

## Width Types

::: faran.types.BoundaryWidths
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.BoundaryWidthsDescription
    options:
      show_root_heading: true
      heading_level: 3
