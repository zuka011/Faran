# Comfort Costs

Comfort costs penalize aggressive control inputs to produce smoother driving behavior.

## Control Smoothing

Penalizes the difference between consecutive control inputs[@Liniger2015]:

```python
from faran.numpy import costs
from numtypes import array

smoothing = costs.comfort.control_smoothing(
    weights=array([5.0, 20.0, 5.0], shape=(3,)),
)
```

The weight vector has one entry per control dimension. Higher weights penalize rapid changes in that control dimension more aggressively.

## Control Effort

Penalizes the magnitude of control inputs[@Williams2017]:

```python
effort = costs.comfort.control_effort(
    weights=array([0.1, 0.5, 0.1], shape=(3,)),
)
```

## API Reference

See the [costs API reference](../../api/costs.md) for full signatures.

\bibliography
