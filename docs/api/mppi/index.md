---
status: draft
---

# mppi

MPPI (Model Predictive Path Integral) is a sampling-based [MPC](../../guide/concepts/mpc.md) algorithm for trajectory planning[@Williams2015][@Williams2016]. For background on trajectory planning and MPC, see the [guide](../../guide/concepts/planning.md).

## Algorithm

Given a dynamical model $f$, cost function $\mathbf{J}$, sampling distribution $\boldsymbol{\mathcal{Z}}$, time horizon $T$, rollout count $M$, planning period $P$, and temperature $\lambda$:

1. Sample $M$ control perturbations around the nominal sequence: $\mathbf{u}_m = \text{sample}(\mathbf{u}, \boldsymbol{\mathcal{Z}})$
2. Simulate each through the dynamics: $\mathbf{x}_m = \text{simulate}(\mathbf{u}_m, \mathsf{x}_0, f)$
3. Evaluate costs: $\mathbf{J}_m = \mathbf{J}(\mathbf{u}_m, \mathbf{x}_m)$
4. Compute the optimal control via softmax-weighted averaging:

$$
w_m = \frac{1}{\eta} \exp\!\left(-\frac{1}{\lambda}(\mathbf{J}_m - \mathbf{J}_{\min})\right), \quad \eta = \sum_{m=1}^{M} \exp\!\left(-\frac{1}{\lambda}(\mathbf{J}_m - \mathbf{J}_{\min})\right)
$$

$$
\mathbf{u}_{\text{opt}} = \text{filter}\!\left(\sum_{m=1}^{M} w_m \, \mathbf{u}_m\right)
$$

After computing $\mathbf{u}_{\text{opt}}$, the nominal sequence is updated, the first $P$ actions are executed, and the sequence is shifted forward with zero-padding.

| Symbol | Meaning |
|--------|---------|
| $T$ | Time horizon (planning steps) |
| $M$ | Rollout count |
| $P$ | Planning interval (steps executed before replanning) |
| $\lambda$ | Temperature (softmax sharpness) |
| $\mathbf{J}_{\min}$ | Minimum cost across all rollouts |
| $w_m$ | Importance weight for rollout $m$ |

## Configuration Levels

Three factory functions provide increasing levels of abstraction.

### `mppi.base`

Assembles a planner from a [model](../model/index.md), [cost function](../costs/index.md), and [sampler](../sampler/index.md).

```python
from faran.numpy import mppi, model, sampler, costs, types
import numpy as np

planner = mppi.base(
    model=model.bicycle.dynamical(time_step_size=0.1, wheelbase=2.5),
    cost_function=costs.combined(...),
    sampler=sampler.gaussian(
        standard_deviation=np.array([0.5, 0.05]),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `DynamicalModel` | — | Dynamics for simulating rollouts |
| `cost_function` | `CostFunction` | — | Scores each rollout at every time step |
| `sampler` | `Sampler` | — | Generates $M$ control perturbations |
| `planning_interval` | `int` | `1` | Steps executed before replanning ($P$) |
| `update_function` | `UpdateFunction \| None` | `UseOptimalControlUpdate` | How `nominal` is updated from `optimal` |
| `padding_function` | `PaddingFunction \| None` | `ZeroPadding` | Fills shifted-out time steps |
| `filter_function` | `FilterFunction \| None` | `NoFilter` | Post-processes the optimal sequence (e.g. [Savitzky-Golay](filter.md)) |

### `mppi.augmented`

Composes separate physical and virtual models into a single augmented planner. Used when the state space needs extension (e.g. adding a path parameter for [MPCC](../../guide/concepts/mpcc.md)).

```python
from faran.numpy import mppi, model, sampler, costs, types
import numpy as np

setup = mppi.augmented(
    models=(physical_model, virtual_model),
    samplers=(physical_sampler, virtual_sampler),
    cost=costs.combined(...),
    state=types.augmented.state,
    state_sequence=types.augmented.state_sequence,
    state_batch=types.augmented.state_batch,
    input_batch=types.augmented.control_input_batch,
)
planner = setup.mppi
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `tuple[DynamicalModel, DynamicalModel]` | — | Physical and virtual dynamics |
| `samplers` | `tuple[Sampler, Sampler]` | — | Separate samplers per sub-model |
| `cost` | `CostFunction` | — | Cost over the augmented state |
| `state` | type | — | Augmented state constructor |
| `state_sequence` | type | — | Augmented state sequence constructor |
| `state_batch` | type | — | Augmented state batch constructor |
| `input_batch` | type | — | Augmented control input batch constructor |
| `planning_interval` | `int` | `1` | Steps executed before replanning |
| `update_function` | `UpdateFunction \| None` | `UseOptimalControlUpdate` | Nominal update strategy |
| `padding_function` | `PaddingFunction \| None` | `ZeroPadding` | Fills shifted-out steps |
| `filter_function` | `FilterFunction \| None` | `NoFilter` | Post-processing filter |

### `mppi.mpcc`

Highest-level factory for [MPCC](../../guide/concepts/mpcc.md) path following. Automatically constructs the augmented state space with a virtual path parameter $\phi$, and configures [contouring, lag, and progress costs](../costs/tracking.md).

Returns a `NamedTuple` of `(mppi, model, contouring_cost, lag_cost)`.

```python
from faran.numpy import mppi, model, sampler, costs, types, extract, filters
from numtypes import array

planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
    model=model.bicycle.dynamical(
        time_step_size=0.1, wheelbase=2.5,
        speed_limits=(0.0, 15.0), steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    sampler=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
    costs=(
        costs.comfort.control_smoothing(weights=array([5.0, 20.0, 5.0], shape=(3,))),
    ),
    reference=reference,
    position_extractor=extract.from_physical(lambda states: states.positions),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `DynamicalModel` | — | Physical dynamics model |
| `sampler` | `Sampler` | — | Sampler for the physical control inputs |
| `reference` | `Trajectory` | — | Reference path for MPCC tracking |
| `position_extractor` | `PositionExtractor` | — | Extracts $(x, y)$ from the augmented state batch |
| `costs` | `tuple[CostFunction, ...]` | `()` | Additional cost terms appended to tracking costs |
| `config` | `MpccConfig.Partial \| None` | See defaults | Weight and virtual-state configuration |
| `planning_interval` | `int` | `1` | Steps executed before replanning |
| `update_function` | `UpdateFunction \| None` | `UseOptimalControlUpdate` | Nominal update strategy |
| `padding_function` | `PaddingFunction \| None` | `ZeroPadding` | Fills shifted-out steps |
| `filter_function` | `FilterFunction \| None` | `NoFilter` | Post-processing filter |

#### MPCC Configuration Defaults

The `config` dict is deep-merged with these defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `weights.contouring` | `20.0` | Contouring cost weight $k_c$ |
| `weights.lag` | `10.0` | Lag cost weight $k_l$ |
| `weights.progress` | `500.0` | Progress cost weight $k_p$ |
| `virtual.state_limits` | `(0.0, reference.path_length)` | Bounds on $\phi$ |
| `virtual.velocity_limits` | `(0.0, 15.0)` | Bounds on $\dot{\phi}$ |
| `virtual.sampling_standard_deviation` | `1.0` | $\sigma$ for the virtual sampler |
| `virtual.sampling_seed` | `0` | RNG seed for the virtual sampler |
| `virtual.periodic` | `False` | Wrap $\phi$ at limits (for looped paths) |

## Planning Loop

```python
state = initial_state
nominal = initial_nominal_input

for step in range(max_steps):
    control = planner.step(
        temperature=50.0, nominal_input=nominal, initial_state=state,
    )
    state = dynamics.step(inputs=control.optimal, state=state)
    nominal = control.nominal
```

`planner.step()` returns a `Control` with:

| Field | Description |
|-------|-------------|
| `control.optimal` | Weighted-average control sequence for this step |
| `control.nominal` | Shifted sequence for warm-starting the next step |
| `control.debug.trajectory_weights` | Softmax weights $w_m$ for all $M$ rollouts |

## Mppi Protocol

::: faran.types.Mppi
    options:
      show_root_heading: true
      heading_level: 3

\bibliography
