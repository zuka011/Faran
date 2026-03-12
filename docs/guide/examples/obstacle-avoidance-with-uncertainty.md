# Obstacle Avoidance with Uncertainty

Four moving obstacles with covariance propagation and a mean-variance risk metric. Uses [state estimation](../estimation/index.md) to track obstacles, propagates uncertainty through motion predictions, and evaluates collision risk using sampled obstacle poses.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/04_obstacle_avoidance_with_uncertainty.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:visualize"
    ```

=== "Result"

    <iframe src="../../../visualizations/mpcc-simulation/doc-dynamic-obstacles-uncertain.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py"
    ```
