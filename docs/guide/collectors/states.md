# State Collector

Records the state at each planning step.

## Usage

```python
from faran import collectors

planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states,
)
```

The `transformer` converts individual states into a state sequence. Use the appropriate type for your model:

| Model | Transformer |
|-------|-------------|
| Augmented (MPCC) | `types.augmented.state_sequence.of_states` |
| Bicycle | `types.bicycle.state_sequence.of_states` |
| Unicycle | `types.unicycle.state_sequence.of_states` |
| Simple | `types.simple.state_sequence.of_states` |

For augmented states, specify both physical and virtual transformers:

```python
planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states(
        physical=types.bicycle.state_sequence.of_states,
        virtual=types.simple.state_sequence.of_states,
    ),
)
```

## Risk Collector

Records risk metric values over time:

```python
planner = collectors.risk.decorating(planner)
```

## API Reference

See the [collectors API reference](../../api/collectors.md) for full signatures.
