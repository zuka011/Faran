# Computational Framework

All computations operate on 3D tensors with a consistent layout.

## Tensor Shapes

| Shape | Meaning |
|-------|---------|
| $(T, D_x, M)$ | State batch — $T$ time steps, $D_x$ state dimensions, $M$ rollouts |
| $(T, D_u, M)$ | Control batch — same layout for control inputs |
| $(T, M)$ | Cost array — one scalar per rollout per time step |

MPPI sums costs over $T$ to get a total cost per rollout, then computes softmax weights to combine all $M$ samples.

## Extractors

Extractors decouple cost functions from specific state representations. A cost function never accesses state arrays directly — it asks an extractor for the values it needs.

This lets the same cost function work with different models:

```python
from faran.numpy import extract

# For augmented states: extract position from the physical sub-state
position = extract.from_physical(lambda states: states.positions)

# For augmented states: extract path parameter from the virtual sub-state
path_param = extract.from_virtual(lambda states: states.array[:, 0, :])
```

## Processing Pipeline

```
Sampler → Perturbations → Model.step (×T) → State batch → Cost function → Costs → Softmax → Optimal control
```

Each component is independent: you can swap any model, cost function, or sampler without changing the others.
