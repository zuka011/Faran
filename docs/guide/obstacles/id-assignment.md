# ID Assignment

When obstacles are detected per-frame without persistent IDs, the library matches detections to tracked obstacles using the Hungarian algorithm on position distances.

## Usage

```python
from faran.numpy import obstacles

id_assignment = obstacles.id_assignment.hungarian(
    position_extractor=obstacle_position_extractor,
    cutoff=5.0,
)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `position_extractor` | Extracts obstacle positions from the state representation. |
| `cutoff` | Maximum matching distance (meters). Detections beyond this get new IDs. |

## How It Works

1. Computes a pairwise distance matrix between current detections and last known positions
2. Solves the assignment via `scipy.optimize.linear_sum_assignment` (NumPy) or equivalent (JAX)
3. Pairs with distance $\leq$ `cutoff` are matched; unmatched detections receive new IDs

The assignment function is called automatically by the running history when new observations are appended.

## API Reference

See the [obstacles API reference](../../api/obstacles.md) for full signatures.
