from typing import Protocol

from faran.types.array import Array
from faran.types.predictors import ObstacleStatesHistory
from faran.types.models.integrator.common import (
    IntegratorState,
    IntegratorStateSequence,
    IntegratorStateBatch,
    IntegratorControlInputSequence,
    IntegratorControlInputBatch,
)

from jaxtyping import Float


class NumPyIntegratorState(IntegratorState, Protocol):
    @property
    def array(self) -> Float[Array, " D_x"]:
        """Returns the underlying NumPy array representing the integrator state."""
        ...


class NumPyIntegratorStateSequence(IntegratorStateSequence, Protocol):
    @property
    def array(self) -> Float[Array, "T D_x"]:
        """Returns the underlying NumPy array representing the integrator state sequence."""
        ...


class NumPyIntegratorStateBatch(IntegratorStateBatch, Protocol):
    @property
    def array(self) -> Float[Array, "T D_x M"]:
        """Returns the underlying NumPy array representing the integrator state batch."""
        ...


class NumPyIntegratorControlInputSequence(IntegratorControlInputSequence, Protocol):
    @property
    def array(self) -> Float[Array, "T D_u"]:
        """Returns the underlying NumPy array representing the integrator control input
        sequence.
        """
        ...


class NumPyIntegratorControlInputBatch(IntegratorControlInputBatch, Protocol):
    @property
    def array(self) -> Float[Array, "T D_u M"]:
        """Returns the underlying NumPy array representing the integrator control input
        batch.
        """
        ...


class NumPyIntegratorObstacleStatesHistory(ObstacleStatesHistory, Protocol):
    @property
    def array(self) -> Float[Array, "T D_o K"]:
        """Returns the obstacle history as a NumPy array."""
        ...
