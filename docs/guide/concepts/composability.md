---
status: draft
---

# Composability

Components can be freely combined because of **decoupled semantic meaning** and **data encapsulation**.

## Decoupled Components

All components in Faran are designed to be as decoupled as possible. For example, cost functions don't assume any particular state representation, and instead ask for extractors or other components to retrieve the necessary information. Sure, you could use structural subtyping for the same purpose, but with so many components and possible combinations, passing explicit "glue" components is cleaner.

Despite the flexibility, the library is statically typed in a way that ensures incompatible components will produce type errors. For example, you'll get a type error if you try to directly use a unicycle model state estimator together with a bicycle model for motion prediction (since some necessary state information would be missing for motion prediction).
 