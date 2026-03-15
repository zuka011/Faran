---
status: draft
---

# types

Core protocols and type definitions used throughout Faran. For dimension and shape conventions, see [Conventions](../../guide/concepts/conventions.md).

## Namespace Access

=== "NumPy"

    ```python
    from faran.numpy import types

    positions = types.positions(x=..., y=...)
    ```

=== "JAX"

    ```python
    from faran.jax import types

    positions = types.positions(x=..., y=...)
    ```

## State Protocols

::: faran.types.State
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.StateSequence
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.StateBatch
    options:
      show_root_heading: true
      heading_level: 3

## Control Protocols

::: faran.types.ControlInputSequence
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ControlInputBatch
    options:
      show_root_heading: true
      heading_level: 3

## Trajectory & Path Types

- [`Trajectory`](../trajectory/index.md) — see [trajectory](../trajectory/index.md)

::: faran.types.PathParameters
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.Positions
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ReferencePoints
    options:
      show_root_heading: true
      heading_level: 3

## Cost Types

::: faran.types.Costs
    options:
      show_root_heading: true
      heading_level: 3

- [`CostFunction`](../costs/index.md) — see [costs](../costs/index.md)

## Boundary Types

::: faran.types.BoundaryDistance
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3

## Model Types

- [`DynamicalModel`](../model/index.md) — see [model](../model/index.md)

## MPPI Types

- [`Mppi`](../mppi/index.md) — see [mppi](../mppi/index.md)

::: faran.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3
