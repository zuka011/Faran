# Comfort Metric

Reports lateral acceleration and jerk relative to the reference trajectory.

## Usage

```python
from faran import metrics

comfort_metric = metrics.comfort(
    reference=reference,
    time_step_size=0.1,
    position_extractor=position_extractor,
)
```

## Results

```python
result = registry.get(comfort_metric)
result.lateral_acceleration  # (T,) — lateral acceleration at each step
result.lateral_jerk          # (T,) — lateral jerk at each step
```

## API Reference

See the [metrics API reference](../../api/metrics.md) for full signatures.
