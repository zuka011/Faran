from typing import Protocol, Self, Any

from faran.types.array import Array
from faran.types.mppi.common import (
    State,
    StateSequence,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Costs,
    CostFunction,
    Sampler,
    UpdateFunction,
    PaddingFunction,
    FilterFunction,
)

from jaxtyping import Float


class NumPyState(State, Protocol):
    @property
    def array(self) -> Float[Array, " D_x"]:
        """Returns the underlying NumPy array representing the state."""
        ...


class NumPyStateSequence[StateBatchT = Any](StateSequence[StateBatchT], Protocol):
    @property
    def array(self) -> Float[Array, "T D_x"]:
        """Returns the underlying NumPy array representing the state sequence."""
        ...


class NumPyStateBatch(StateBatch, Protocol):
    @property
    def array(self) -> Float[Array, "T D_x M"]:
        """Returns the underlying NumPy array representing the state batch."""
        ...


class NumPyControlInputSequence(ControlInputSequence, Protocol):
    def similar(self, *, array: Float[Array, "T D_u"]) -> Self:
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...

    @property
    def array(self) -> Float[Array, "T D_u"]:
        """Returns the underlying NumPy array representing the control input sequence."""
        ...


class NumPyControlInputBatch(ControlInputBatch, Protocol):
    @property
    def array(self) -> Float[Array, "T D_u M"]:
        """Returns the underlying NumPy array representing the control input batch."""
        ...


class NumPyCosts(Costs, Protocol):
    def similar(self, *, array: Float[Array, "T M"]) -> Self:
        """Creates new costs similar to this one but with the given array as its data."""
        ...

    @property
    def array(self) -> Float[Array, "T M"]:
        """Returns the underlying NumPy array representing the costs."""
        ...


class NumPyDynamicalModel[
    StateT,
    StateSequenceT,
    StateBatchT,
    InputSequenceT,
    InputBatchT,
](
    DynamicalModel[StateT, StateSequenceT, StateBatchT, InputSequenceT, InputBatchT],
    Protocol,
): ...


class NumPySampler[InputSequenceT, InputBatchT](
    Sampler[InputSequenceT, InputBatchT], Protocol
): ...


class NumPyCostFunction[InputBatchT, StateBatchT, CostsT](
    CostFunction[InputBatchT, StateBatchT, CostsT], Protocol
): ...


class NumPyUpdateFunction[InputSequenceT](UpdateFunction[InputSequenceT], Protocol): ...


class NumPyPaddingFunction[NominalT, PaddingT](
    PaddingFunction[NominalT, PaddingT], Protocol
): ...


class NumPyFilterFunction[InputSequenceT](FilterFunction[InputSequenceT], Protocol): ...
