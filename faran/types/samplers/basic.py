from typing import Protocol

from faran.types.array import Array

from jaxtyping import Float


class NumPyControlInputBatchCreator[
    InputBatchT,
](Protocol):
    def __call__(self, *, array: Float[Array, "T D_u M"]) -> InputBatchT:
        """Creates a control input batch from the given array."""
        ...
