from typing import Protocol, runtime_checkable

from faran.types.array import Array, DataType

from jaxtyping import Float


class Error(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        """Returns the error as a NumPy array."""
        ...


class Preference(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "M"]:
        """Returns the preference as a NumPy array."""
        ...


class PositionExtractor[StateBatchT, PositionsT](Protocol):
    def __call__(self, states: StateBatchT, /) -> PositionsT:
        """Extracts (x, y) positions from a batch of states."""
        ...


@runtime_checkable
class TrajectoryPreferenceProvider[StateBatchT, PreferenceT](Protocol):
    def __call__(self, states: StateBatchT, /) -> PreferenceT:
        """Provides a preference score for each state sequence in the batch, where higher
        scores indicate more preferred trajectories."""
        ...


class ContouringCost[InputBatchT, StateBatchT, ErrorT = Error](Protocol):
    def error(self, *, states: StateBatchT) -> ErrorT:
        """Computes the contouring error for the given states."""
        ...


class LagCost[InputBatchT, StateBatchT, ErrorT = Error](Protocol):
    def error(self, *, states: StateBatchT) -> ErrorT:
        """Computes the lag error for the given states."""
        ...
