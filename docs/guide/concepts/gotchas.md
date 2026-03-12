# Gotchas

Known limitations and non-obvious behaviors. If you hit something not listed here, please [open an issue](https://gitlab.com/risk-metrics/faran/-/issues).

## JAX Cost Functions Are Not All Differentiable

Some cost functions in JAX (e.g., collision costs) are not automatically differentiable via `jax.grad`. Until a gradient-based planning algorithm is added, this is not a priority.

**Workaround:** Use the sampling-based [MPPI planner](../planners/mppi.md).

## SAT Distance Assumes 2D Cartesian Space

The SAT-based distance estimator (`distance.sat`) works with convex polygons in 2D cartesian space only.

**Workaround:** Use circle-based distance (`distance.circles`), which works with any shape approximable by circles.

## Collision Cost Weight Ordering

The `weight` factor in the collision cost is applied **after** the risk metric computation (`weight * metric(cost)`), not before. This matters for non-positively-homogeneous risk metrics like `mean_variance(gamma > 0)` where `risk(w * c) ≠ w * risk(c)`.

## Backend Mixing

All components in a pipeline must use the same backend. Mixing `faran.numpy` and `faran.jax` objects produces errors.

## JIT Warm-up

With the JAX backend, add a warm-up call before timing-critical code to trigger JIT compilation.
