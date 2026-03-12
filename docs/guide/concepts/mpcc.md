# MPCC: Model Predictive Contouring Control

MPCC is an MPC formulation for path following[@Liniger2015]. It introduces a virtual path parameter $\phi$ that moves independently along the reference trajectory, and decomposes tracking error into two components:

- **Contouring error** $e_c$ — perpendicular distance to the path (lateral deviation)
- **Lag error** $e_l$ — distance along the path behind the reference point (longitudinal deviation)

$$
e_c = \sin(\theta_\phi)(x - x_\phi) - \cos(\theta_\phi)(y - y_\phi)
$$

$$
e_l = -\cos(\theta_\phi)(x - x_\phi) - \sin(\theta_\phi)(y - y_\phi)
$$

A progress cost pushes $\phi$ forward while contouring and lag costs pull the vehicle toward the reference point.

## Augmented State

MPCC augments the physical state with a virtual component:

| | Variables | Meaning |
|---|---|---|
| Physical state | $x, y, \theta, v$ | Vehicle pose and speed |
| Virtual state | $\phi$ | Arc-length progress along the reference |
| Physical controls | $a, \delta$ | Acceleration, steering |
| Virtual control | $\dot\phi$ | Path velocity |

Both the physical and virtual dynamics are simulated together. The `mppi.mpcc` factory handles this composition automatically.

## Factory

```python
from faran.numpy import mppi

planner, model, contouring, lag = mppi.mpcc(
    model=...,
    sampler=...,
    reference=reference,
    contouring_weight=50.0,
    lag_weight=100.0,
    progress_weight=1000.0,
    ...
)
```

## Cost Balance

The balance between the three [tracking costs](../costs/tracking.md) determines tracking behavior:

- High contouring weight → tight lateral tracking
- High lag weight → keeps up with the reference point
- High progress weight → fast traversal, potentially cutting corners

## API Reference

See the [MPPI API reference](../../api/mppi.md) for `mppi.mpcc` signatures.

\bibliography
