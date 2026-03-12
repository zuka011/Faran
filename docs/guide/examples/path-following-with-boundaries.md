# Path Following with Boundaries

Adds a fixed-width corridor (2.5 m each side) to the basic example. The [boundary cost](../costs/safety.md#boundary) steers the vehicle away from corridor edges.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/02_path_following_with_boundaries.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:visualize"
    ```

=== "Result"

    <iframe src="../../../visualizations/mpcc-simulation/doc-path-following-with-boundary.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py"
    ```
