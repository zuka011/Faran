from typing import Protocol, Self, Any

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

from jaxtyping import Array as JaxArray, Float


class JaxState(State, Protocol):
    @property
    def array(self) -> Float[JaxArray, " D_x"]:
        """Returns the underlying JAX array representing the state."""
        ...


class JaxStateSequence[StateBatchT = Any](StateSequence[StateBatchT], Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_x"]:
        """Returns the underlying JAX array representing the state sequence."""
        ...


class JaxStateBatch(StateBatch, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        """Returns the underlying JAX array representing the state batch."""
        ...


class JaxControlInputSequence(ControlInputSequence, Protocol):
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self:
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


class JaxControlInputBatch(ControlInputBatch, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array representing the control input batch."""
        ...


class JaxCosts(Costs, Protocol):
    def similar(self, *, array: Float[JaxArray, "T M"]) -> Self:
        """Creates new costs similar to this one but with the given array as its data."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        """Returns the underlying JAX array representing the costs."""
        ...


class JaxDynamicalModel[
    StateT,
    StateSequenceT,
    StateBatchT,
    InputSequenceT,
    InputBatchT,
](
    DynamicalModel[StateT, StateSequenceT, StateBatchT, InputSequenceT, InputBatchT],
    Protocol,
): ...


class JaxSampler[InputSequenceT, InputBatchT](
    Sampler[InputSequenceT, InputBatchT], Protocol
): ...


class JaxCostFunction[InputBatchT, StateBatchT, CostsT](
    CostFunction[InputBatchT, StateBatchT, CostsT], Protocol
): ...


class JaxUpdateFunction[InputSequenceT](UpdateFunction[InputSequenceT], Protocol): ...


class JaxPaddingFunction[NominalT, PaddingT](
    PaddingFunction[NominalT, PaddingT], Protocol
): ...


class JaxFilterFunction[InputSequenceT](FilterFunction[InputSequenceT], Protocol): ...
