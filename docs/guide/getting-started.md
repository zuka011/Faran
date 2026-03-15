---
reading_time: true
---

# Getting Started

One of the goals of Faran is to get you to work on the problem you care about as quickly as possible. That means not wasting time on reinventing the wheel, and since trajectory planners for simple path following have been set up a hundred times, let's see if we can get you a working planner in 15 minutes.

## Installation

```bash
pip install faran          # NumPy + JAX (CPU)
pip install faran[cuda]    # JAX with GPU support (Linux)
```

Requires Python 3.13+. 

The visualizer is a separate package (optional, but recommended dependency):

```bash
pip install faran-visualizer
```

## Your First Planner

We'll build a planner that follows a reference path using a [kinematic bicycle model](../api/model/bicycle.md).

A simple way to configure a path for this purpose is to use the [`mppi.mpcc()`](../api/mppi/index.md) factory. It assembles an [MPPI](../api/mppi/index.md#mppi) planner with contouring, lag, and progress costs for path following using the [MPCC formulation](concepts/mpcc.md) of an [MPC](concepts/mpc.md) problem.

For this walkthrough, you may also want to install [tqdm](https://github.com/tqdm/tqdm) to make waiting a bit more exciting:

```bash
pip install tqdm
```

### Setup

We first need a reference path. For simplicity, let's just create one by connecting a few dots in 2D space:

```python
--8<-- "docs/examples/01_basic_path_following.py:reference"

# These will also come in handy later
--8<-- "docs/examples/01_basic_path_following.py:constants"
```

The waypoints in the above example trace something like an "S" curve. You can use your *mind's eye* to see it... or wait for the visualization at the end of this tutorial. The `path_length` parameter can be omitted, but it helps to have a fixed "parameterization" length for MPCC. A consistent parameterization length makes tuning the planner's progress cost weight easier (more on that [later](#how-mpcc-works)).

Now we can configure the planner to follow that path:

```python
--8<-- "docs/examples/01_basic_path_following.py:setup"
```

The `mppi.mpcc()` factory returns four objects:

| Object            | What it is                                                                      |
|-------------------|---------------------------------------------------------------------------------|
| `planner`         | The MPPI planner. You call `.step()` to get the planned control inputs          |
| `augmented_model` | The full MPCC system dynamics                                                   |
| `contouring_cost` | Contouring cost, useful for error [evaluation metrics](../api/metrics/index.md) |
| `lag_cost`        | Lag cost, also for error [evaluation metrics](../api/metrics/index.md)          |

We use a vanilla [Gaussian sampler](../api/sampler/gaussian.md) to generate control perturbations (although you are by no means limited to that) and a [Savitzky-Golay filter](../api/mppi/filter.md#savitzky-golay-filter) to make the vehicle passengers a little less nauseous. A peculiar part may be the `position_extractor` parameter, which tells the planner what "position" means for the setup. Since Faran components try to be as flexible as possible, sometimes an extra bit of *wiring* code needs to be written to connect the components together. Faran may have more magic 🪄 in the future, but for now there's no free lunch.

??? note "What's with the weird dict passed as a config?"

    You can use the `config` parameter to configure the preset components that `mppi.mpcc()` uses. It's a dict, because having to import another config class would be clumsy + it's a typed dict, so you still get type safety and intellisense.

After the planner is created, we also set up a couple of collectors and metrics. The collectors in Faran decorate their target components and spy on the data flowing through them. At the end we also have a registry to keep all the collectors and metrics in one place for easy access. 

### Simulation Loop

Let's run the planner in a loop. Each call to `planner.step()` will return the "plan" for the current step, which includes:

- `control.optimal` — the best control sequence for this step
- `control.nominal` — the shifted control sequence used to warm-start the next step

```python
--8<-- "docs/examples/01_basic_path_following.py:loop"
```

Before we actually call `run`, we'll also need this function to collect the simulation results for visualization and error analysis:

```python
--8<-- "docs/examples/01_basic_path_following.py:result"

--8<-- "docs/examples/01_basic_path_following.py:extract"
```

Now run the whole thing like so:

```python
planner, augmented_model, registry, error_metric = create()
result = run(planner, augmented_model, registry, error_metric)
```

### Result

After 100 steps the vehicle will have tracked the reference path. If you have [`faran-visualizer`](visualizer.md) installed, you can now generate an interactive visualization:

```python
# Call this function with the `Result` object returned by `run()`
--8<-- "docs/examples/01_basic_path_following.py:visualize"
```

```python
# For example, like this:
import asyncio

asyncio.run(visualize(result))
```

Voilà! This produces a standalone HTML file you can open in any browser:

<iframe src="../../visualizations/mpcc-simulation/doc-basic-path-following.html"
        width="100%" height="700px" frameborder="0"></iframe>

??? note "Full example"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
    ```

You can also check out this example on Binder → [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zurabmu/faran/main?filepath=notebooks/01_basic_path_following.ipynb){ .binder-badge }

## How MPCC Works

MPCC augments the system state with a virtual path parameter $\phi$ that tracks progress along a [reference trajectory](../api/trajectory/index.md):

| Component | State               | Controls       |
|-----------|---------------------|----------------|
| Physical  | $[x, y, \theta, v]$ | $[a, \delta]$  |
| Virtual   | $[\phi]$            | $[\dot{\phi}]$ |

Three [costs](../api/costs/index.md) components make sure ego follows the reference path:

- **Contouring** — penalizes lateral deviation from the reference point $\phi$
- **Lag** — penalizes longitudinal offset between $\phi$ and the vehicle's projection
- **Progress** — rewards forward motion along the path ($\dot\phi > 0$)

The balance between these three costs determines tracking behavior. High contouring weight keeps the vehicle close to the path; high progress weight makes it drive faster.

!!! tip "Need more control?"

    For use with custom models, additional costs, or different samplers, see [MPPI](../api/mppi/index.md#mppi).

## Next Steps

- **[Concepts](concepts/mpcc.md)** — Understand the MPCC formulation of an optimization problem.
- **[MPPI Planning](../api/mppi/index.md#mppi)** — Learn how exactly MPPI solves MPC problems like MPCC.
- **[Cost Functions](../api/costs/index.md)** — Add safety, comfort, or custom objectives beyond basic path tracking.
- **[Obstacles](../api/obstacles/index.md)** — Avoid moving obstacles in the environment by predicting their motion.
- **[Examples](examples/index.md)** — See complete example scenarios with interactive visualizations.
