---
status: draft
---

# Samplers

Samplers generate the control perturbation batches $\{\boldsymbol{\epsilon}^{(m)}\}_{m=1}^{M}$ evaluated by the [MPPI planner](../mppi/index.md) at each iteration.

| Sampler | Temporal Structure | Sequence Type |
|---------|-------------------|---------------|
| [Gaussian](gaussian.md) | i.i.d. per timestep | Pseudo-random |
| [Halton Spline](halton.md) | Correlated via cubic splines | Quasi-random (low discrepancy)[@Halton1960] |

## Sampler Protocol

::: faran.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3

\bibliography
