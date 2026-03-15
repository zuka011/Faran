---
status: draft
---

# Planner Factories

Factory functions and concrete classes for the [MPPI](index.md) planner. See the [parent page](index.md) for the algorithm description, equations, and configuration-level overview.

## NumPyMppi

::: faran.mppi.basic.NumPyMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - step

## JaxMppi

::: faran.mppi.accelerated.JaxMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - step

## MPCC Factory

::: faran.mpcc.basic.NumPyMpccMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.mpcc.accelerated.JaxMpccMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Supporting Types

### Control

The return type of `planner.step()`.

| Field | Type | Description |
|-------|------|-------------|
| `optimal` | `ControlInputSequence` | Weighted-average optimal control sequence |
| `nominal` | `ControlInputSequence` | Shifted sequence for warm-starting the next step |
| `debug` | `DebugData` | Contains `trajectory_weights` ($w_m$ for each rollout) |

::: faran.types.Control
    options:
      show_root_heading: true
      heading_level: 4

### Update Functions

| Class | Behavior |
|-------|----------|
| `UseOptimalControlUpdate` | Sets nominal to the optimal input (default) |
| `NoUpdate` | Keeps the nominal input unchanged |

::: faran.mppi.common.UseOptimalControlUpdate
    options:
      show_root_heading: true
      heading_level: 4

::: faran.mppi.common.NoFilter
    options:
      show_root_heading: true
      heading_level: 4
