---
status: draft
---

# Backends

Faran exposes compatible APIs for two backends. You can switch by changing the import statements:

```python
# NumPy — prototyping, debugging
from faran.numpy import mppi, model, sampler, costs, trajectory, boundary, types

# JAX — GPU acceleration, JIT compilation
from faran.jax import mppi, model, sampler, costs, trajectory, boundary, types
```

Compatible means that all backends speak the language of NumPy. That means NumPy arrays can be used for configuration everywhere, and all outputs can be converted to NumPy arrays with `np.asarray()` (see [Output Conversion](#output-conversion)). Also, all backends support the API that the NumPy backend defines, but specific backends may have additional features.

## Consistency

All components in a pipeline must use the same backend. Mixing `faran.numpy` and `faran.jax` objects will produce errors.

## Output Conversion

All results or inputs can be converted to NumPy arrays via `np.asarray()`:

```python
import numpy as np

numpy_array = np.asarray(jax_control.optimal)
```

This is convenient if you want to serialize data, perform computations that are not performance-sensitive, or whatever else you might use NumPy for.
