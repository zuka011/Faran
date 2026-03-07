# MPPI Planning

This page covers the practical details of configuring and tuning an MPPI planner[@Williams2015][@Williams2016][@Williams2017]: factory functions, temperature selection, filtering, and sampler seeding. For the algorithmic background, see [Core Concepts](concepts.md).

## Factory Functions

| Factory          | Use case                                                                         |
|------------------|----------------------------------------------------------------------------------|
| `mppi.base`      | Single model, custom cost function                                               |
| `mppi.augmented` | Augmented states (e.g., physical + virtual) with separate models and samplers    |
| `mppi.mpcc`      | MPCC path following — wires up contouring, lag, and progress costs automatically |

### `mppi.base`

```python
from faran.numpy import mppi, model, sampler, costs, types, filters
from numtypes import array

planner = mppi.base(
    model=model.bicycle.dynamical(
        time_step_size=0.1, wheelbase=2.5,
        speed_limits=(0.0, 15.0), steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    cost_function=cost,
    sampler=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create, seed=42,
    ),
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
)
```

### `mppi.augmented`

Composes multiple sub-models (e.g., bicycle + integrator) and samplers into one planner:

```python
from faran.states import AugmentedModel, AugmentedSampler

planner, augmented_model = mppi.augmented(
    models=(bicycle_model, integrator_model),
    samplers=(physical_sampler, virtual_sampler),
    cost=cost,
    state=types.augmented.state,
    state_sequence=types.augmented.state_sequence,
    state_batch=types.augmented.state_batch,
    input_batch=types.augmented.control_input_batch,
)
```

### `mppi.mpcc`

Sets up MPCC path following in one call — creates the augmented model, contouring/lag/progress costs, and the planner:

```python
planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
    model=model.bicycle.dynamical(...),
    sampler=sampler.gaussian(...),
    reference=reference,
    position_extractor=extract.from_physical(
        lambda states: states.positions
    ),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)
```

The returned `contouring_cost` and `lag_cost` objects can be used to inspect tracking errors during simulation (see [Metrics](metrics.md)).

## Running the Planner

```python
control = planner.step(
    temperature=50.0,
    nominal_input=nominal,
    initial_state=state,
)

# control.optimal  — weighted-average control sequence
# control.nominal  — shifted nominal for next iteration (warm start)
```

### Simulation Loop

```python
state = initial_state
nominal = initial_nominal

for step in range(max_steps):
    control = planner.step(
        temperature=50.0, nominal_input=nominal, initial_state=state,
    )
    state = model.step(inputs=control.optimal, state=state)
    nominal = control.nominal
```

## Temperature

The temperature $\lambda$ controls the sharpness of the softmax weighting used to combine rollout costs into the optimal control:

$$
w_m = \frac{1}{\eta} \exp\!\left(-\frac{1}{\lambda}\,(J_m - J_{\min})\right)
$$

- **Low** ($\lambda \approx 1$–$10$) — concentrates weight on the lowest-cost samples. More greedy, less diverse. Good when costs are small and well-scaled.
- **High** ($\lambda \approx 50$–$100$) — distributes weight more evenly across samples. More exploration, smoother control. Necessary when costs are large.

### Choosing a Temperature

The right $\lambda$ depends on the **magnitude of your cost values**. If your total costs per rollout are in the range $[100, 10\,000]$, a temperature of $50$ will produce reasonable weights. If costs are in $[0.01, 1.0]$, try $\lambda = 0.1$.

A practical starting point: set $\lambda$ so that the cost difference $(J_m - J_{\min})$ divided by $\lambda$ is roughly in the range $[0.1, 10]$ for most rollouts. If all weights collapse to zero (numerical underflow), increase $\lambda$. If all weights are nearly uniform, decrease it.

## Filtering

A Savitzky-Golay filter smooths the optimal control sequence, reducing jitter from noisy sampling:

```python
from faran.numpy import filters

planner = mppi.base(
    ...,
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
)
```

**`window_length`** must be odd and determines how many time steps are smoothed together. Larger windows produce smoother but less reactive control. **`polynomial_order`** controls fitting flexibility — 3 (cubic) is a good default.

If the planner's control outputs look noisy or oscillatory, try increasing the window length. If the planner reacts too slowly to changes, reduce it.

## Sampler Seeding

Samplers are deterministic given a seed. When using `mppi.augmented`, use different seeds for the physical and virtual samplers to avoid correlated perturbations:

```python
physical_sampler = sampler.gaussian(seed=42, ...)
virtual_sampler = sampler.gaussian(seed=43, ...)
```

\bibliography
