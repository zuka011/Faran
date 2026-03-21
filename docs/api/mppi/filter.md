---
status: draft
---

# Filters

Filters post-process the weighted-average optimal control sequence produced by the [MPPI](index.md) planner. They correspond to the $\text{filter}(\cdot)$ operation in the MPPI algorithm.

## Savitzky-Golay Filter

Fits a low-degree polynomial to successive windows of the optimal control sequence via least squares[@Savitzky1964][@Williams2018], then takes the fitted value at each window center as the smoothed output. Applied independently to each control dimension.

Given a half-window size $w$ and polynomial degree $p$ (with $p < 2w + 1$), the smoothed value at time step $t$ is:

$$
\hat{y}_t = \sum_{j=-w}^{w} c_j \, y_{t+j}
$$

where $\{c_j\}$ are convolution coefficients determined solely by $w$ and $p$, precomputed once.

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `window_length` | `int` | Must be odd | Number of time steps in the smoothing window ($2w + 1$) |
| `polynomial_order` | `int` | $0 \leq p < \text{window\_length}$ | Degree of the fitting polynomial |

```python
from faran.numpy import filters

sg_filter = filters.savgol(window_length=11, polynomial_order=3)
```

Pass to any [MPPI factory](index.md) via `filter_function`:

```python
from faran.numpy import mppi

planner = mppi.base(..., filter_function=sg_filter)
```

::: faran.mppi.savgol.basic.NumPySavGolFilter
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.mppi.savgol.accelerated.JaxSavGolFilter
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## No Filter

Identity filter — returns the optimal input unchanged. This is the default when no `filter_function` is provided.

::: faran.mppi.common.NoFilter
    options:
      show_root_heading: true
      heading_level: 3

\bibliography
