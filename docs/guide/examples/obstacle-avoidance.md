---
reading_time: true
status: draft
---

# Obstacle Avoidance

Three static obstacles sit along the path. This example adds [circle-based collision geometry](../../api/obstacles/index.md#circle-to-circle) and a [collision cost](../../api/costs/safety.md#collision-cost) so the planner navigates around them while staying inside the corridor from [Path Following with Boundaries](path-following-with-boundaries.md).

## Reference Path

A longer path than previous examples (`path_length=70.0`), giving the planner room to maneuver around the obstacles:

```python
--8<-- "docs/examples/03_obstacle_avoidance.py:reference"
```

## Setup

```python
--8<-- "docs/examples/03_obstacle_avoidance.py:setup"
```

The new components compared to [Path Following with Boundaries](path-following-with-boundaries.md):

| Component                                   | What it does                                                                                                                                              |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `create_obstacles.static(...)`              | Places three obstacles at fixed positions with specified headings                                                                                         |
| `predictor.curvilinear(...)`                | Forecasts obstacle motion over the planning horizon using a bicycle obstacle model and a [finite-difference estimator](../../api/estimation/index.md)     |
| `create_obstacles.provider.predicting(...)` | Feeds observation history through the predictor and tracks obstacle identities across time steps via [Hungarian assignment](../../api/obstacles/index.md) |
| `Circles(...)`                              | Defines circle-based collision geometry — three overlapping circles per entity, spaced 0.5 m apart with 0.8 m radius                                      |
| `distance.circles(...)`                     | Computes [signed distance](../../api/obstacles/index.md#circle-to-circle) between the ego and obstacle circle models                                      |
| `costs.safety.collision(...)`               | Penalizes rollouts where the signed distance falls below `distance_threshold` (0.5 m per circle)                                                          |

The ego vehicle and each obstacle are approximated by the same three overlapping circles. The collision cost samples obstacle poses from the predictor's distribution and evaluates the minimum signed distance between these circle sets — rollouts that intrude past `distance_threshold` get penalized with `weight=1500.0`.

A few other values changed from the previous example to accommodate obstacle avoidance:

- **Corridor width**: 2.5 → 3.0 m per side — wider to give the planner room to swerve
- **Rollout count**: 256 → 512 — more samples improve cost estimation when the planner must choose between tight gaps
- **Virtual velocity minimum**: 0.0 → 1.0 — prevents the planner from stalling in front of an obstacle instead of going around it

The `BicyclePredictionCreator` and `NumPyObstaclePositionExtractor` helper classes visible in the full example are wiring code that tells the predictor how to extract poses and positions from obstacle states — similar to the `position_extractor` in earlier examples.

## Simulation Loop

One addition compared to previous examples: the obstacle observer feeds each simulator step into the predictor pipeline.

```python
--8<-- "docs/examples/03_obstacle_avoidance.py:loop"
```

The `obstacle_observer.observe(obstacle_simulator.step())` line is the new piece. Even for static obstacles, the predictor needs observations to build its internal state. In a real system these would come from a perception pipeline rather than a simulator.

To run:

```python
components = create()
result = run(*components)
```

## Result

```python
--8<-- "docs/examples/03_obstacle_avoidance.py:visualize"
```

```python
import asyncio

asyncio.run(visualize(result))
```

The visualization shows three obstacles placed at (20, 2.5), (35, 7.5), and (50, 2.5) along the reference path, with a 3.0 m corridor on each side:

<iframe src="../../../visualizations/mpcc-simulation/doc-static-obstacles.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full example"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py"
    ```

You can also run this example on Binder → [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zurabmu/faran/main?filepath=notebooks/03_obstacle_avoidance.ipynb){ .binder-badge }

## Next Steps

- **[Obstacle Avoidance with Uncertainty](obstacle-avoidance-with-uncertainty.md)** — Handle moving obstacles with uncertain predictions using risk metrics.
- **[Collision Cost](../../api/costs/safety.md#collision-cost)** — Full API reference for collision penalization.
- **[Obstacles](../../api/obstacles/index.md)** — Distance models, obstacle providers, and motion predictors.
