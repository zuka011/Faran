---
reading_time: true
---

# Model Predictive Control

Model Predictive Control (MPC) is a popular optimization-based control strategy[@Camacho2004]. Instead of solving the entire control problem once (e.g. deciding in advance how to drive to the supermarket), MPC solves a series of smaller optimization problems at each time step (e.g. deciding how to take the next turn without crashing into the neighbor's fence). The idea is to find an optimal sequence of control inputs over a finite time horizon, execute a part of that sequence, then replan at the next step using updated information on the system's state and environment.

Given the current state $x_0$, MPC solves the following constrained optimization problem over a horizon of $T$ steps:

$$
\min_{u_0, \dots, u_{T-1}} \; \sum_{t=0}^{T-1} \ell(x_t, u_t) + V(x_T) \\
\text{s.t.} \quad x_{t+1} = f(x_t, u_t), \quad u_t \in \mathcal{U} \quad \forall \, t \in [0, T{-}1]
$$

where:

- $f$ is the dynamics model,
- $\ell$ is the stage cost function,
- $V$ is the terminal cost function, and
- $\mathcal{U}$ is the input constraint set.

!!! note "$J$ vs $\ell$ and $V$"

    Faran does not differentiate between stage and terminal costs. There is only a single cost function $J$ that takes in the full trajectory and control sequence.

!!! note "Mathematical formulation"

    This is just one simplified mathematical formulation of a discrete-time MPC problem over a finite horizon. 

Since the optimization is solved at each step, MPC can adapt to changes in the environment or system state. It can also account for errors, like model mismatch or tracking errors.

## Solving MPC Problems

There are many algorithms for solving the MPC optimization problem, which can be broadly categorized into **gradient-based** and **sampling-based** approaches:

| Approach       | Optimization                | Requirements                     |
|----------------|-----------------------------|----------------------------------|
| Gradient-based | Nonlinear programming (NLP) | Differentiable dynamics and cost |
| Sampling-based | Monte Carlo evaluation      | Forward simulation only          |

**Gradient-based** methods (e.g., iLQR, SQP) compute locally optimal trajectories using derivative information. **Sampling-based** methods (e.g., [MPPI](../../api/mppi/index.md#mppi)) draw many candidate trajectories and combine them through cost-weighted averaging. Sampling-based methods have the advantage that they can handle non-differentiable costs and dynamics, and are easier to parallelize on accelerators.

\bibliography
