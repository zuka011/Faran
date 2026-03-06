# Visualizer

`faran-visualizer` generates standalone HTML files with interactive Plotly charts from simulation results. Each visualization includes an animated vehicle replay and configurable time-series plots.

## Installation

```bash
pip install faran-visualizer
```

Requires **Node.js 18+** at runtime.

## How It Works

1. You build a result object (`MpccSimulationResult` or `Visualizable.SimulationResult`) from your simulation data.
2. You call a visualizer factory (`visualizer.mpcc()` or `visualizer.simulation()`) to create a renderer.
3. The renderer serializes the result to JSON and invokes the TypeScript core to produce a self-contained HTML file.

Output goes to the configured directory (default: current working directory). Each call produces `<key>.json` and `<key>.html`.

```python
from faran_visualizer import configure
configure(output_directory="./results")
```

## MPCC Visualization

For planners created with `mppi.mpcc()`, use `MpccSimulationResult`:

```python
import asyncio
from faran_visualizer import visualizer, MpccSimulationResult, configure

result = MpccSimulationResult(
    reference=reference,
    states=collected_states,
    contouring_errors=errors.contouring,
    lag_errors=errors.lag,
    time_step_size=0.1,
    wheelbase=2.5,
)

configure(output_directory=".")
asyncio.run(visualizer.mpcc()(result, key="my-simulation"))
```

### Required Fields

| Field               | Type            | Description                                 |
|---------------------|-----------------|---------------------------------------------|
| `reference`         | `Trajectory`    | The reference path used by MPCC             |
| `states`            | `StateSequence` | Augmented state history from collectors     |
| `contouring_errors` | `array`         | Contouring errors from `metrics.mpcc_error` |
| `lag_errors`        | `array`         | Lag errors from `metrics.mpcc_error`        |
| `time_step_size`    | `float`         | Simulation time step (seconds)              |
| `wheelbase`         | `float`         | Vehicle wheelbase (meters)                  |

### Optional Fields

| Field                  | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `obstacles`            | Obstacle state history for rendering obstacles                     |
| `obstacle_forecasts`   | Predicted obstacle trajectories                                    |
| `boundary`             | Corridor boundary for rendering road edges                         |
| `controls`             | Control input history                                              |
| `risks`                | Risk metric values over time                                       |
| `optimal_trajectories` | Best trajectory at each step                                       |
| `nominal_trajectories` | Nominal (warm-started) trajectory at each step                     |
| `vehicle_width`        | Width of the rendered vehicle (default: proportional to wheelbase) |
| `max_contouring_error` | Cap for contouring error plot y-axis                               |
| `max_lag_error`        | Cap for lag error plot y-axis                                      |
| `network`              | Road network overlay                                               |

## Generic Visualization

For simulations that don't use the MPCC factory:

```python
from faran_visualizer import visualizer, Visualizable

result = Visualizable.SimulationResult.create(
    info=Visualizable.SimulationInfo(
        path_length=100.0, time_step=0.1, wheelbase=2.5,
    ),
    reference=Visualizable.ReferenceTrajectory(x=ref_x, y=ref_y),
    ego=Visualizable.Ego(
        x=ego_x, y=ego_y, heading=ego_heading, path_parameter=ego_progress,
    ),
)

asyncio.run(visualizer.simulation()(result, key="my-simulation"))
```

## Custom Plots

Attach additional time-series plots with optional bounds and uncertainty bands:

```python
from faran_visualizer import Plot

speed_plot = Plot.Additional(
    id="speed",
    name="Vehicle Speed",
    series=[Plot.Series(label="Speed", values=speeds, color="blue")],
    y_axis_label="Speed (m/s)",
    upper_bound=Plot.Bound(values=15.0, label="Max Speed"),
)
```

Pass additional plots to the `MpccSimulationResult` or `Visualizable.SimulationResult` via the `additional_plots` field.

## What the Output Shows

The HTML file contains an interactive dashboard with a visualization of the vehicle's trajectory, the reference path, and any obstacles. Time-series plots show contouring and lag errors, control inputs, risk metrics, or any custom data you provide. All plots are interactive (Plotly): pan, zoom, and hover for exact values.

For API signatures and detailed options, see the [Visualizer API reference](../api/visualizer.md).
