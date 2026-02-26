from typing import Protocol

from faran.types.predictors import ObstacleStatesHistory
from faran.types.models.integrator.common import (
    IntegratorState,
    IntegratorStateSequence,
    IntegratorStateBatch,
    IntegratorControlInputSequence,
    IntegratorControlInputBatch,
)

from jaxtyping import Array as JaxArray, Float


class JaxIntegratorState(IntegratorState, Protocol):
    @property
    def array(self) -> Float[JaxArray, " D_x"]:
        """Returns the underlying JAX array representing the integrator state."""
        ...


class JaxIntegratorStateSequence(IntegratorStateSequence, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_x"]:
        """Returns the underlying JAX array representing the integrator state sequence."""
        ...


class JaxIntegratorStateBatch(IntegratorStateBatch, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        """Returns the underlying JAX array representing the integrator state batch."""
        ...


class JaxIntegratorControlInputSequence(IntegratorControlInputSequence, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the integrator control input sequence."""
        ...


class JaxIntegratorControlInputBatch(IntegratorControlInputBatch, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array representing the integrator control input batch."""
        ...


class JaxIntegratorObstacleStatesHistory(ObstacleStatesHistory, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle history as a JAX array."""
        ...
