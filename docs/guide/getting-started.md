# Getting Started

This guide walks you through installing Faran, building a planner, and running your first simulation loop.

## Installation

```bash
pip install faran          # NumPy + JAX (CPU)
pip install faran[cuda]    # JAX with GPU support (Linux)
```

Requires Python 3.13+. The visualizer is a separate optional package:

```bash
pip install faran-visualizer
```

## Your First Planner

The fastest way to a working planner is `mppi.mpcc()`. It assembles an MPPI planner with contouring, lag, and progress costs for path following using the [MPCC formulation](concepts.md#mpcc-model-predictive-contouring-control).

### Setup

```python
--8<-- "docs/examples/01_basic_path_following.py:setup"
```

This creates four objects:

- **`planner`** — the MPPI planner, ready to call `.step()`
- **`augmented_model`** — the combined physical + virtual dynamics model
- **`contouring_cost`** and **`lag_cost`** — cost objects you can use later for [evaluation metrics](metrics.md)

### Simulation Loop

Run the planner in a loop. Each iteration samples control sequences, evaluates their costs, and returns the weighted-average optimal control:

```python
--8<-- "docs/examples/01_basic_path_following.py:loop"
```

`control.optimal` is the best control sequence from this step. `control.nominal` is the shifted sequence used as the sampling center for the next step (warm-starting).

??? note "Full example"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
    ```

## What `mppi.mpcc()` Sets Up

MPCC augments the vehicle state with a virtual path parameter $\phi$ that tracks progress along a reference trajectory:

| Component | State | Controls |
|-----------|-------|----------|
| Physical | $[x, y, \theta, v]$ | $[a, \delta]$ |
| Virtual | $[\phi]$ | $[\dot{\phi}]$ |

Three costs drive path following:

- **Contouring** — penalizes lateral deviation from the reference
- **Lag** — penalizes longitudinal offset between $\phi$ and the vehicle's projection
- **Progress** — rewards forward motion along the path ($\dot\phi > 0$)

The balance between these three costs determines tracking behavior. High contouring weight keeps the vehicle close to the path; high progress weight makes it drive faster.

For manual assembly with custom models, additional costs, or mixed samplers, see [Core Concepts](concepts.md).

## Next Steps

- [Core Concepts](concepts.md) — MPPI algorithm and MPCC formulation
- [MPPI Planning](mppi.md) — Temperature, filtering, seeding
- [Cost Function Design](costs.md) — Tracking, safety, and comfort objectives
- [Obstacle Handling](obstacles.md) — Collision avoidance with distance functions
- [Examples](examples.md) — Complete scenarios with interactive visualizations
