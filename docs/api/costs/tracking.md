---
status: draft
---

# Tracking Costs

MPCC path-following objectives[@Liniger2015]. Require extractors mapping augmented states to positions and path parameters — provided automatically by `mppi.mpcc(...)`.

## Contouring Cost

Lateral (orthogonal) deviation from the reference trajectory:

$$
J_c = k_c \, e_c^2, \quad e_c = \sin(\theta_\phi)(x - x_\phi) - \cos(\theta_\phi)(y - y_\phi)
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference` | `Trajectory` | Reference path |
| `path_parameter_extractor` | extractor | Maps state → $\phi$ |
| `position_extractor` | extractor | Maps state → $(x, y)$ |
| `weight` | `float` | $k_c$ |

::: faran.costs.basic.NumPyContouringCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - error

::: faran.costs.accelerated.JaxContouringCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - error

## Lag Cost

Longitudinal (tangential) deviation from the reference point:

$$
J_l = k_l \, e_l^2, \quad e_l = -\cos(\theta_\phi)(x - x_\phi) - \sin(\theta_\phi)(y - y_\phi)
$$

Same parameters as contouring cost with weight $k_l$.

::: faran.costs.basic.NumPyLagCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - error

::: faran.costs.accelerated.JaxLagCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - error

## Progress Cost

Rewards forward motion along the reference path:

$$
J_p = -k_p \, \dot{\phi} \, \Delta t
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `path_velocity_extractor` | extractor | Maps control → $\dot{\phi}$ |
| `time_step_size` | `float` | $\Delta t$ |
| `weight` | `float` | $k_p$ |

::: faran.costs.basic.NumPyProgressCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.accelerated.JaxProgressCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

\bibliography
