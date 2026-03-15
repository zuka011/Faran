---
status: draft
---

# Comfort Costs

## Control Smoothing Cost

Penalizes rate of change between consecutive control inputs[@Liniger2015]:

$$
J_s = \sum_{d} w_d \, (\mathbf{u}_{t,d} - \mathbf{u}_{t-1,d})^2
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | array of shape $(D_u,)$ | Per-dimension weight $w_d$ |

```python
from faran.numpy import costs
import numpy as np

smoothing = costs.comfort.control_smoothing(
    weights=np.array([5.0, 20.0, 5.0]),
)
```

::: faran.costs.basic.NumPyControlSmoothingCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.accelerated.JaxControlSmoothingCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Control Effort Cost

Penalizes control input magnitudes[@Williams2017]:

$$
J_n = \sum_{d} w_d \, \mathbf{u}_{t,d}^2
$$

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | array of shape $(D_u,)$ | Per-dimension weight $w_d$ |

```python
effort = costs.comfort.control_effort(
    weights=np.array([0.1, 0.5, 0.1]),
)
```

::: faran.costs.basic.NumPyControlEffortCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.accelerated.JaxControlEffortCost
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

\bibliography
