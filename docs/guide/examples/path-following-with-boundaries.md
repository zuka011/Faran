---
reading_time: true
status: draft
---

# Path Following with Boundaries

This example extends [Getting Started](../getting-started.md) by confining the vehicle to a fixed-width corridor. A [boundary cost](../../api/costs/safety.md#boundary-cost) steers it away from the corridor edges, and a [control smoothing cost](../../api/costs/comfort.md) dampens the sharper corrections that corridor constraints can cause.

## Setup

The reference path here is a sharp turn instead of the S-curve from Getting Started — the exact shape doesn't matter, but the corridor effect is easier to see when there's a potentional temptation to cut corners.

Three new components are added via the `costs` parameter on `mppi.mpcc()`, which is how you extend the planner beyond its built-in contouring, lag, and progress costs:

```python
--8<-- "docs/examples/02_path_following_with_boundaries.py:setup"
```

| Component                              | What it does                                                     |
|----------------------------------------|------------------------------------------------------------------|
| `boundary.fixed_width(...)`            | Defines a corridor 1.5 m wide on each side of the reference path |
| `costs.comfort.control_smoothing(...)` | Penalizes abrupt changes across the augmented control dimensions |
| `costs.safety.boundary(...)`           | Penalizes rollouts that come within 1.0 m of the corridor edge   |

Each cost in the `costs` tuple is evaluated for every MPPI rollout and added to the total. The boundary cost only activates within the `distance_threshold`, so it stays quiet when the vehicle is safely inside the corridor — but its high `weight` (1000.0) ensures it dominates when the vehicle gets close to the edge. The smoothing cost helps here because corridor constraints can cause abrupt steering corrections that the [Savitzky-Golay filter](../../api/mppi/filter.md#savitzky-golay-filter) alone doesn't fully absorb. The middle weight (`20.0`) is highest because steering oscillations are the most noticeable artifact.

The model, sampler, filter, and MPCC weights are identical to [Getting Started](../getting-started.md).

## Simulation Loop

The loop body is unchanged from [Getting Started](../getting-started.md). The boundary cost is already baked into the planner's cost function at setup time, so no extra work is needed here:

```python
--8<-- "docs/examples/02_path_following_with_boundaries.py:loop"
```

To run:

```python
planner, augmented_model, registry, error_metric, corridor = create()
result = run(planner, augmented_model, registry, error_metric, corridor)
```

## Result

```python
--8<-- "docs/examples/02_path_following_with_boundaries.py:visualize"
```

```python
import asyncio

asyncio.run(visualize(result))
```

The visualization shows the reference path with a 1.5 m corridor on each side:

<iframe src="../../../visualizations/mpcc-simulation/doc-path-following-with-boundary.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full example"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py"
    ```

You can also run this example on Binder → [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zurabmu/faran/main?filepath=notebooks/02_path_following_with_boundaries.ipynb){ .binder-badge }

## Next Steps

- **[Obstacle Avoidance](obstacle-avoidance.md)** — Add static obstacles the planner must navigate around.
- **[Boundary Cost](../../api/costs/safety.md#boundary-cost)** — Full API reference for boundary penalties.
