---
status: draft
---

# Conventions

Dimension symbols and array shapes used throughout the API.

## Index Dimensions

| Symbol | Meaning                         |
|--------|---------------------------------|
| $T$    | Time horizon (planning steps)   |
| $M$    | Rollout count (samples)         |
| $K$    | Obstacle count                  |
| $N$    | Prediction samples per obstacle |

## State and Input Dimensions

| Symbol | Meaning                  | Bicycle                          | Unicycle                      | Integrator |
|--------|--------------------------|----------------------------------|-------------------------------|------------|
| $D_x$  | State dimension          | 4 $(x, y, \theta, v)$            | 3 $(x, y, \theta)$            | $n$        |
| $D_u$  | Control input dimension  | 2 $(a, \delta)$                  | 2 $(v, \omega)$               | $n$        |
| $D_o$  | Obstacle state dimension | 6 $(x, y, \theta, v, a, \delta)$ | 5 $(x, y, \theta, v, \omega)$ | $n$        |

## Array Shapes

| Type                        | Shape            | Notes                              |
|-----------------------------|------------------|------------------------------------|
| `State`                     | $(D_x,)$         | Single state vector                |
| `StateSequence`             | $(T, D_x)$       | Trajectory over time               |
| `StateBatch`                | $(T, D_x, M)$    | Batched rollouts                   |
| `ControlInputSequence`      | $(T, D_u)$       | Single control sequence            |
| `ControlInputBatch`         | $(T, D_u, M)$    | Batched control sequences          |
| `ObstacleStatesForTimeStep` | $(D_o, K)$       | $K$ obstacles at a single step     |
| `ObstacleStates`            | $(T, D_o, K)$    | $K$ obstacles over $T$ steps       |
| `SampledObstacleStates`     | $(T, D_o, K, N)$ | $N$ prediction samples             |
| `Cost`                      | $(T, M)$         | Cost for each rollout at each step |
| Others...                   | ...              | ...                                |
