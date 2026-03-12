# Tracking Costs

Used in the [MPCC](../concepts/mpcc.md) formulation[@Liniger2015]. Tracking costs decompose path-following error into contouring (lateral), lag (longitudinal), and progress components.

## Contouring

Penalizes lateral deviation from the reference path:

$$
J_c = k_c \cdot e_c^2
$$

```python
from faran.numpy import costs, extract

contouring = costs.tracking.contouring(
    reference=reference,
    path_parameter_extractor=extract.from_virtual(path_parameter),
    position_extractor=extract.from_physical(
        lambda states: states.positions
    ),
    weight=50.0,
)
```

## Lag

Penalizes longitudinal deviation from the reference point:

$$
J_l = k_l \cdot e_l^2
$$

```python
lag = costs.tracking.lag(
    reference=reference,
    path_parameter_extractor=extract.from_virtual(path_parameter),
    position_extractor=extract.from_physical(
        lambda states: states.positions
    ),
    weight=100.0,
)
```

## Progress

Rewards forward motion along the path. Without this, $\phi$ would stay at zero:

$$
J_p = -k_p \cdot \dot\phi \cdot \Delta t
$$

```python
progress = costs.tracking.progress(
    path_velocity_extractor=extract.from_virtual(path_velocity),
    time_step_size=0.1,
    weight=1000.0,
)
```

## API Reference

See the [costs API reference](../../api/costs.md) for full signatures.

\bibliography
