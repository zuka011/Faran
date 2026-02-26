from typing import Protocol, overload

from faran.types.array import Array

from jaxtyping import Float

import numpy as np


class HistoryWithArray(Protocol):
    @property
    def array(self) -> Float[Array, "T D_o K"]:
        """Returns the history as a Numpy array."""
        ...


class EstimationFilter(Protocol):
    @overload
    def __call__(self, array: Float[Array, " K"], /) -> Float[Array, " K"]:
        """Filters out invalid estimates from the given array."""
        ...

    @overload
    def __call__(self, array: Float[Array, "D_o K"], /) -> Float[Array, "D_o K"]:  # pyright: ignore[reportOverlappingOverload]
        """Filters out invalid estimates from the given array."""
        ...


def invalid_obstacle_filter_from(
    history: HistoryWithArray, *, check_recent: int
) -> "EstimationFilter":
    """Returns a filter for invalid estimates based on the given history.

    Args:
        history: The history to check for invalid estimates.
        check_recent: The number of most recent time steps to check for invalid estimates.
    """
    assert check_recent > 0, (
        "At least one time step must be checked for invalid estimates"
    )

    invalid = np.any(np.isnan(history.array[-check_recent:]), axis=(0, 1))

    def filter_invalid(array: Array) -> Array:
        return np.where(invalid, np.nan, array)

    return filter_invalid
