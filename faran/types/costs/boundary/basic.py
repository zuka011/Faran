from typing import Protocol
from dataclasses import dataclass

from faran.types.array import Array, DataType, jaxtyped
from faran.types.costs.boundary.common import BoundaryDistanceExtractor

from jaxtyping import Float


class NumPyBoundaryDistanceExtractor[StateBatchT, DistanceT](
    BoundaryDistanceExtractor[StateBatchT, DistanceT], Protocol
): ...


@jaxtyped
@dataclass(frozen=True)
class NumPyBoundaryDistance:
    _array: Float[Array, "T M"]

    @staticmethod
    def create(*, array: Float[Array, "T M"]) -> "NumPyBoundaryDistance":
        """Creates a NumPy boundary distance from the given array."""
        return NumPyBoundaryDistance(array)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        return self._array

    @property
    def horizon(self) -> int:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._array.shape[1]

    @property
    def array(self) -> Float[Array, "T M"]:
        return self._array
