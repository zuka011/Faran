---
reading_time: true
---

# Computational Framework

To keep Faran performant, all components are designed to perform computations in batches. Keeping in mind that the underlying numerical libraries (Numpy, JAX) automatically vectorize/parallelize batched operations, this makes performance optimizations easier, but it also means there are a lot of high-dimensional arrays floating around. If you're implementing custom components, the following conventions used in Faran could be helpful to follow:

- Arrays are always typed using [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for readability, even if the shapes are not strictly enforced at runtime. In critical code paths, [beartype](https://beartype.readthedocs.io/en/latest/) is used for runtime type checking.
- If a time dimension exists, it is always the first dimension. This makes it easier to implement cost functions that operate on states and inputs from different time steps (e.g. you would write `inputs[t] - inputs[t-1]` to get the input difference between two consecutive time steps.)
- If a state/control dimension exists, it is always the second dimension. The batching dimensions (e.g. rollout, samples, obstacles) are placed at the end, since these are least likely to be accessed individually.

For example, MPPI expects a cost function to accept `Float[Array, "T D_x M"]` states and `Float[Array, "T D_u M"]` inputs, where `T` is the time dimension, `D_x` and `D_u` are the state and control dimensions, and `M` is the number of rollouts/samples. The returned costs should have shape `Float[Array, "T M"]` (i.e. a cost for each time step and rollout).
