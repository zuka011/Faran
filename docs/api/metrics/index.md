---
status: draft
---

# metrics

Post-simulation evaluation metrics operating on data recorded by [collectors](../collectors/index.md). Metrics are registered in a `MetricRegistry` and computed lazily on first access.

## Available Metrics

| Metric | Factory | Result Type | Description |
|--------|---------|-------------|-------------|
| [Collision](#collision) | `metrics.collision(...)` | `CollisionMetricResult` | Minimum distances and collision flags |
| [Task Completion](#task-completion) | `metrics.task_completion(...)` | `TaskCompletionMetricResult` | Goal arrival, stretch ratio, progress |
| [MPCC Error](#mpcc-error) | `metrics.mpcc_error(...)` | `MpccErrorMetricResult` | Contouring and lag tracking errors |
| [Constraint Violation](#constraint-violation) | `metrics.constraint_violation(...)` | `ConstraintViolationMetricResult` | Boundary distance and violations |
| [Comfort](#comfort) | `metrics.comfort(...)` | `ComfortMetricResult` | Lateral acceleration and jerk |

## Setup

```python
from faran import collectors, metrics

state_collector = collectors.states.decorating(planner, transformer=...)
obstacle_collector = collectors.obstacle_states.decorating(observer)

registry = metrics.registry(
    collision_metric := metrics.collision(
        distance_threshold=0.5, distance=distance_extractor,
    ),
    task_metric := metrics.task_completion(
        reference=reference, distance_threshold=2.0,
        time_step_size=0.1, position_extractor=position_extractor,
    ),
    collectors=collectors.registry(state_collector, obstacle_collector),
)
```

## Collision

Evaluates minimum ego–obstacle distances and detects collisions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance_threshold` | `float` | Distance at or below which a collision is registered |
| `distance` | `DistanceExtractor` | Distance computation (circles or SAT) |

| Result Field | Type | Description |
|--------------|------|-------------|
| `distances` | `(T, V) array` | Distance per time step per vehicle part |
| `min_distances` | `(V,) array` | Minimum distance over time |
| `collisions` | `(T, V) bool array` | Collision flags |
| `collision_detected` | `bool` | Any collision at any time |

## Task Completion

Tracks progress along the reference trajectory and goal arrival.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference` | `Trajectory` | — | Reference trajectory |
| `distance_threshold` | `float` | — | Distance within which the goal is considered reached |
| `time_step_size` | `float` | — | Simulation time step size |
| `position_extractor` | `PositionExtractor` | — | Extracts positions from states |
| `lap_detection_threshold` | `float` | `0.5` | Fraction of path length drop that triggers a new lap |

| Result Field | Type | Description |
|--------------|------|-------------|
| `completed` | `bool` | Whether the goal was reached |
| `completion_time` | `float` | Seconds to reach goal (`inf` if not reached) |
| `stretch` | `float` | Ratio of traversed distance to optimal distance |
| `completed_part` | `float` | Fraction of reference completed (includes laps) |

## MPCC Error

Reports contouring and lag errors for [MPCC](../../guide/concepts/mpcc.md) controllers. Requires the cost objects returned by `mppi.mpcc(...)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `contouring` | `ContouringCost` | Contouring cost from MPCC setup |
| `lag` | `LagCost` | Lag cost from MPCC setup |

| Result Field | Type | Description |
|--------------|------|-------------|
| `contouring` | `(T,) array` | Contouring error at each time step |
| `lag` | `(T,) array` | Lag error at each time step |
| `max_contouring` | `float` | Peak absolute contouring error |
| `max_lag` | `float` | Peak absolute lag error |

## Constraint Violation

Reports boundary distances and flags violations where the vehicle leaves the corridor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | `Trajectory` | Reference trajectory |
| `boundary` | `BoundaryDistanceExtractor` | Boundary distance computation |
| `position_extractor` | `PositionExtractor` | Extracts positions from states |

| Result Field | Type | Description |
|--------------|------|-------------|
| `lateral_deviations` | `(T,) array` | Lateral deviation from reference |
| `boundary_distances` | `(T,) array` | Signed distance to nearest boundary |
| `violations` | `(T,) bool array` | Violation flags |
| `violation_detected` | `bool` | Any violation occurred |

## Comfort

Reports lateral acceleration and jerk derived from lateral deviations via finite differences.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | `Trajectory` | Reference trajectory |
| `time_step_size` | `float` | Simulation time step size |
| `position_extractor` | `PositionExtractor` | Extracts positions from states |

| Result Field | Type | Description |
|--------------|------|-------------|
| `lateral_acceleration` | `(T,) array` | Lateral acceleration at each step |
| `lateral_jerk` | `(T,) array` | Lateral jerk at each step |

## Live Evaluation

Metrics recompute automatically when new data arrives via collectors:

```python
for step in range(horizon):
    state_collector.step(temperature=50.0, nominal_input=nominal, initial_state=state)
    obstacle_collector.observe(obstacle_states)

    if registry.get(collision_metric).collision_detected:
        break
```
