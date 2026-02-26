from typing import Protocol, Self, Any
from dataclasses import dataclass

from faran.types.array import Array, DataType

from jaxtyping import Float


class State(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, " D_x"]:
        """Returns the state as a NumPy array."""
        ...

    @property
    def dimension(self) -> int:
        """Returns the dimension of the state."""
        ...


class StateSequence[StateBatchT = Any](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x"]:
        """Returns the state sequence as a NumPy array."""
        ...

    def batched(self) -> StateBatchT:
        """Returns the state sequence as a batch with a single rollout."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the state sequence."""
        ...

    @property
    def dimension(self) -> int:
        """State dimension."""
        ...


class StateBatch(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x M"]:
        """Returns the states as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the state batch."""
        ...

    @property
    def dimension(self) -> int:
        """State dimension."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts in the batch."""
        ...


class ControlInputSequence(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u"]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> int:
        """Control input dimension."""
        ...


class ControlInputBatch(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u M"]:
        """Returns the control inputs as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control input batch."""
        ...

    @property
    def dimension(self) -> int:
        """Control input dimension."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts in the batch."""
        ...


class DynamicalModel[StateT, StateSequenceT, StateBatchT, InputSequenceT, InputBatchT](
    Protocol
):
    def simulate(self, inputs: InputBatchT, initial_state: StateT) -> StateBatchT:
        """Simulates the dynamical model over the given control inputs starting from the
        provided initial state."""
        ...

    def step(self, inputs: InputSequenceT, state: StateT) -> StateT:
        """Simulates a single time step of the dynamical model given the control input and current
        state."""
        ...

    def forward(self, inputs: InputSequenceT, state: StateT) -> StateSequenceT:
        """Simulates the dynamical model over the given control input sequence starting from the
        provided initial state."""
        ...

    @property
    def time_step_size(self) -> float:
        """Time step size of the dynamical model."""
        ...


class Costs(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        """Returns the costs as a NumPy array."""
        ...

    def zero(self) -> Self:
        """Returns zero costs similar to this one."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the costs."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts the costs correspond to."""
        ...


class CostFunction[InputBatchT, StateBatchT, CostsT](Protocol):
    def __call__(self, *, inputs: InputBatchT, states: StateBatchT) -> CostsT:
        """Computes the cost for each time step and rollout."""
        ...


class Sampler[InputSequenceT, InputBatchT](Protocol):
    def sample(self, *, around: InputSequenceT) -> InputBatchT:
        """Samples a batch of control input sequences around the given nominal input."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts the sampler generates."""
        ...


class Weights(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, " M"]:
        """Returns the weights as a NumPy array."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts the weights correspond to."""
        ...


@dataclass(kw_only=True, frozen=True)
class DebugData[WeightsT]:
    trajectory_weights: WeightsT


@dataclass(frozen=True)
class Control[InputSequenceT, WeightsT]:
    optimal: InputSequenceT
    nominal: InputSequenceT
    debug: DebugData[WeightsT]


class Mppi[StateT, InputSequenceT, WeightsT = Any](Protocol):
    def step(
        self,
        *,
        temperature: float,
        nominal_input: InputSequenceT,
        initial_state: StateT,
    ) -> Control[InputSequenceT, WeightsT]:
        """Runs one iteration of the MPPI algorithm to compute the next optimal and nominal
        control sequences.
        """
        ...


class UpdateFunction[InputSequenceT](Protocol):
    def __call__(
        self, *, nominal_input: InputSequenceT, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        """Updates the nominal control input sequence based on the optimal control input
        sequence.
        """
        ...


class PaddingFunction[NominalT, PaddingT](Protocol):
    def __call__(self, *, nominal_input: NominalT, padding_size: int) -> PaddingT:
        """Generates padding values for the shifted nominal control input sequence."""
        ...


class FilterFunction[InputSequenceT](Protocol):
    def __call__(self, *, optimal_input: InputSequenceT) -> InputSequenceT:
        """Filters the optimal control input after it is computed."""
        ...
