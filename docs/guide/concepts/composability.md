---
status: draft
---

# Composability

Components can be freely combined because of **decoupled semantic meaning** and **data encapsulation**.

## Decoupled Components

Cost functions, samplers, and models are independent of each other. A collision cost does not know whether the ego uses a bicycle or unicycle model — it only asks an extractor for positions and headings.

This means:

- The same collision cost works with any dynamics model
- Samplers are unaware of what model the perturbations will drive
- Cost functions compose via `costs.combined(...)` without knowing about each other

## Extractors

Extractors bridge the gap between generic cost functions and specific state representations:

```python
from faran.numpy import extract

# Works regardless of which model produces the states
position = extract.from_physical(lambda states: states.positions)
heading = extract.from_physical(lambda states: states.headings)
```

## Data Encapsulation

State and control types wrap raw arrays with semantic meaning. Instead of accessing `array[:, 0, :]` directly, use named properties:

```python
states.positions   # (T, 2, M) — x and y
states.headings    # (T, M) — heading angle
states.speeds      # (T, M) — speed (bicycle only)
```

This prevents indexing errors and makes code self-documenting. Type constructors validate shapes at creation time.
 