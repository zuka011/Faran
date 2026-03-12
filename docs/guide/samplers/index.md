# Samplers

Samplers generate control input perturbations around a nominal sequence. At each MPPI step, the sampler produces $M$ perturbed control sequences that the planner evaluates via rollout simulation.

| Sampler | Smoothness | Coverage | Cost |
|---------|-----------|----------|------|
| [Gaussian](gaussian.md) | Independent per time step | Pseudo-random | Cheapest |
| [Halton Spline](halton.md) | Temporally correlated via splines | Quasi-random (low discrepancy) | Slightly more |

Start with Gaussian. Switch to Halton if you see noisy control outputs or want better exploration with fewer rollouts.

## Standard Deviation

The standard deviation controls the exploration radius around the nominal control sequence. Too small → planner gets stuck in local optima. Too large → most samples are infeasible.

For a bicycle model, typical starting values:

- **Acceleration:** 0.3–1.0 (depends on limits)
- **Steering:** 0.02–0.1 (smaller because steering has tighter limits)

Rule of thumb: set to 10–30% of the control limit range.

## API Reference

See the [sampler API reference](../../api/sampler.md) for full signatures.
