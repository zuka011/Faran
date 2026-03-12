# Obstacle Avoidance

Three static obstacles along the path. Demonstrates [circle-based distance](../obstacles/distance.md#circle-to-circle), [collision cost](../costs/safety.md#collision), and a fixed-width corridor. The planner swerves around obstacles while staying inside the corridor.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/03_obstacle_avoidance.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:visualize"
    ```

=== "Result"

    <iframe src="../../../visualizations/mpcc-simulation/doc-static-obstacles.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py"
    ```
