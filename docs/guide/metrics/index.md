# Evaluation Metrics

Evaluation metrics measure planning performance after (or during) a simulation. They operate on data collected by [collectors](../collectors/index.md).

## Setup

Wrap the planner and obstacle observer with collectors, then register the metrics:

```python
from faran import collectors, metrics

mppi_collector = collectors.states.decorating(
    planner,
    transformer=types.simple.state_sequence.of_states,
)

obstacle_collector = collectors.obstacle_states.decorating(observer)

registry = metrics.registry(
    collision_metric := metrics.collision(
        distance_threshold=0.5,
        distance=distance_extractor,
    ),
    task_metric := metrics.task_completion(
        reference=reference,
        distance_threshold=2.0,
        time_step_size=0.1,
        position_extractor=lambda s: types.positions(x=s.array[:, 0], y=s.array[:, 1]),
    ),
    collectors=collectors.registry(mppi_collector, obstacle_collector),
)
```

## Available Metrics

| Metric | Page | Measures |
|--------|------|----------|
| [Collision](collision.md) | Minimum distances, collision detection |
| [Task Completion](task-completion.md) | Goal reached, completion time, stretch |
| [MPCC Error](mpcc-error.md) | Contouring and lag error |
| [Constraint Violation](constraint-violation.md) | Boundary violations |
| [Comfort](comfort.md) | Lateral acceleration and jerk |

## Live Evaluation

Metrics recompute automatically when new data arrives:

```python
for step in range(horizon):
    mppi_collector.step(
        temperature=50.0, nominal_input=nominal, initial_state=state,
    )
    obstacle_collector.observe(obstacle_states)

    if registry.get(collision_metric).collision_detected:
        print("Collision!")
        break
```

## API Reference

See the [metrics API reference](../../api/metrics.md) for full signatures.
