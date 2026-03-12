# Task Completion Metric

Tracks progress along the reference trajectory and goal arrival.

## Usage

```python
from faran import metrics

task_metric = metrics.task_completion(
    reference=reference,
    distance_threshold=2.0,
    time_step_size=0.1,
    position_extractor=lambda s: types.positions(x=s.array[:, 0], y=s.array[:, 1]),
)
```

## Results

```python
result = registry.get(task_metric)
result.completed        # bool — True if goal reached
result.completion_time  # float — seconds to reach goal (inf if not reached)
result.stretch          # float — ratio of traveled distance to optimal
result.completed_part   # float — fraction of reference completed (0 to 1+)
```

`completed_part` can exceed 1.0 for looped trajectories, counting additional laps.

## API Reference

See the [metrics API reference](../../api/metrics.md) for full signatures.
