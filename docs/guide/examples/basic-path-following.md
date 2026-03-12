# Basic Path Following

A [bicycle model](../models/bicycle.md) following an S-curve using contouring, lag, and progress costs. Introduces the core planning loop and [MPCC](../concepts/mpcc.md) path tracking.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/01_basic_path_following.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:visualize"
    ```

=== "Result"

    <iframe src="../../../visualizations/mpcc-simulation/doc-basic-path-following.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
    ```
