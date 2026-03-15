---
status: draft
---

# Single Integrator Model

$D$-dimensional single integrator where each state is driven directly by a velocity input:

$$
\mathbf{x}_{t+1} = \operatorname{clip}\!\bigl(\mathbf{x}_t + \operatorname{clip}(\mathbf{u}_t,\; v_{\min}, v_{\max}) \cdot \Delta t,\; s_{\min}, s_{\max}\bigr)
$$

When `periodic=True`, the state wraps around at the limits instead of clamping. Used internally for the MPCC virtual state (path parameter $\phi$); see [MPCC configuration](../mppi/index.md).

| Component | Variables | Description |
|-----------|-----------|-------------|
| State | $\mathbf{x} \in \mathbb{R}^{D}$ | Position (arbitrary dimension) |
| Controls | $\mathbf{u} \in \mathbb{R}^{D}$ | Velocity commands |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_step_size` | `float` | — | Integration step $\Delta t$ (seconds) |
| `state_limits` | `tuple[float, float] \| None` | `None` | Clamp or wrap bounds on $\mathbf{x}$ |
| `velocity_limits` | `tuple[float, float] \| None` | `None` | Clamp bounds on $\mathbf{u}$ |
| `periodic` | `bool` | `False` | Wrap state at limits instead of clamping |

```python
from faran.numpy import model

integrator = model.integrator.dynamical(
    time_step_size=0.1,
    velocity_limits=(0.0, 15.0),
    periodic=False,
)
```

::: faran.models.integrator.basic.NumPyIntegratorModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create

::: faran.models.integrator.accelerated.JaxIntegratorModel
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create
