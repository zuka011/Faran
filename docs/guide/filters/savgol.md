# Savitzky-Golay Filter

A polynomial-fitting filter that smooths the optimal control sequence, reducing jitter from noisy sampling.

## Usage

```python
from faran.numpy import mppi, filters

planner = mppi.base(
    ...,
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `window_length` | Must be odd. Number of time steps smoothed together. Larger = smoother but less reactive. |
| `polynomial_order` | Fitting flexibility. 3 (cubic) is a good default. |

## Tuning

- Control outputs look noisy or oscillatory → increase `window_length`
- Planner reacts too slowly to changes → decrease `window_length`

## API Reference

See the [MPPI API reference](../../api/mppi.md) for the filter parameter in the planner factory.
