---
status: draft
---

# visualizer

Interactive HTML visualizations for simulation results. Distributed as a separate package. For setup and usage, see the [Visualizer guide](../../guide/visualizer.md).

## Installation

```bash
pip install faran-visualizer
```

## Usage

```python
from faran_visualizer import visualizer

mpcc_viz = visualizer.mpcc(
    output="simulation",
    output_directory="./output",
)
```

## MpccSimulationResult

::: faran_visualizer.api.mpcc.MpccSimulationResult
    options:
      show_root_heading: true
      heading_level: 3

## MpccVisualizer

::: faran_visualizer.api.mpcc.MpccVisualizer
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - __call__

## Plot Types

### Plot.Series

::: faran_visualizer.api.simulation.Plot.Series
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Additional

::: faran_visualizer.api.simulation.Plot.Additional
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Bound

::: faran_visualizer.api.simulation.Plot.Bound
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Band

::: faran_visualizer.api.simulation.Plot.Band
    options:
      show_root_heading: true
      heading_level: 3

## Road Network

### Road.Lane

::: faran_visualizer.api.simulation.Road.Lane
    options:
      show_root_heading: true
      heading_level: 3

### Road.Network

::: faran_visualizer.api.simulation.Road.Network
    options:
      show_root_heading: true
      heading_level: 3
