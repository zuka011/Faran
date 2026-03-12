# MPCC Error Metric

Reports contouring and lag error over the trajectory. Requires the cost objects returned by `mppi.mpcc`.

## Usage

```python
planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(...)

error_metric = metrics.mpcc_error(contouring=contouring_cost, lag=lag_cost)
```

## Results

```python
result = registry.get(error_metric)
result.contouring      # (T,) — contouring error at each time step
result.lag             # (T,) — lag error at each time step
result.max_contouring  # float — peak absolute contouring error
result.max_lag         # float — peak absolute lag error
```

## API Reference

See the [metrics API reference](../../api/metrics.md) for full signatures.
