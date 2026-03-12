# Constraint Violation Metric

Reports boundary distances and flags violations where the vehicle leaves the corridor.

## Usage

```python
from faran import metrics

violation_metric = metrics.constraint_violation(
    reference=reference,
    boundary=corridor,
    position_extractor=position_extractor,
)
```

## Results

```python
result = registry.get(violation_metric)
result.lateral_deviations  # (T,) — lateral offset from the reference
result.boundary_distances  # (T,) — signed distance to corridor edge
result.violations          # (T,) — boolean flags where boundary_distance ≤ 0
result.violation_detected  # bool — True if any violation occurred
```

## API Reference

See the [metrics API reference](../../api/metrics.md) for full signatures.
