---
status: draft
---

# Gotchas

Known limitations and non-obvious behaviors. If you hit something not listed here, please [open an issue](https://gitlab.com/risk-metrics/faran/-/issues).

## JAX Cost Functions Are Not All Differentiable

Some cost functions in JAX (e.g., collision costs) are not automatically differentiable via `jax.grad`. Until a gradient-based planning algorithm is added, this is not a priority.

**Workaround:** Use the sampling-based [MPPI planner](../../api/mppi/index.md#mppi).

## SAT Distance Assumes 2D Cartesian Space

The SAT-based distance estimator (`distance.sat`) works with convex polygons in 2D cartesian space only.

**Workaround:** None for now. You can raise [a feature request](https://gitlab.com/risk-metrics/faran/-/issues) though.
