# Boundaries

Boundaries define drivable corridors around a reference trajectory. They produce signed distances that feed into a [boundary cost](../costs/safety.md#boundary).

## Distance Convention

Calling the corridor returns signed distances with shape $(T, M)$:

```python
distances = corridor(states=states)
```

| Value | Meaning |
|-------|---------|
| $d > 0$ | Inside corridor |
| $d = 0$ | On boundary |
| $d < 0$ | Violation |

Two corridor types are available: [fixed width](fixed-width.md) and [variable width](variable-width.md).

## API Reference

See the [boundary API reference](../../api/boundary.md) for full signatures.
