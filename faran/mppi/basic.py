from typing import cast, Any
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    Array,
    DataType,
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyDynamicalModel,
    NumPyCostFunction,
    NumPySampler,
    NumPyUpdateFunction,
    NumPyPaddingFunction,
    NumPyFilterFunction,
    Mppi,
    DebugData,
    Control,
)
from faran.mppi.common import UseOptimalControlUpdate, NoFilter

from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(frozen=True)
class NumPyWeights:
    """Softmax-normalized cost weights used to compute the weighted average over rollouts."""

    _array: Float[Array, " M"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, " M"]:
        return self._array

    @property
    def rollout_count(self) -> int:
        return self._array.shape[0]

    @property
    def array(self) -> Float[Array, " M"]:
        return self._array


class NumPyZeroPadding(
    NumPyPaddingFunction[NumPyControlInputSequence, NumPyControlInputSequence]
):
    """Fills shifted-out time steps with zero control inputs after horizon advancement."""

    def __call__(
        self, *, nominal_input: NumPyControlInputSequence, padding_size: int
    ) -> NumPyControlInputSequence:
        array = np.zeros((padding_size, nominal_input.dimension))

        return nominal_input.similar(array=array)


@dataclass(kw_only=True, frozen=True)
class NumPyMppi[
    StateT: NumPyState,
    StateBatchT: NumPyStateBatch,
    ControlInputSequenceT: NumPyControlInputSequence,
    ControlInputBatchT: NumPyControlInputBatch,
    ControlInputPaddingT: NumPyControlInputSequence = ControlInputSequenceT,
](Mppi[StateT, ControlInputSequenceT, NumPyWeights]):
    """Sampling-based stochastic optimal controller using the information-theoretic MPPI algorithm."""

    planning_interval: int
    model: NumPyDynamicalModel[
        StateT, Any, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ]
    cost_function: NumPyCostFunction
    sampler: NumPySampler[ControlInputSequenceT, ControlInputBatchT]
    update_function: NumPyUpdateFunction[ControlInputSequenceT]
    padding_function: NumPyPaddingFunction[ControlInputSequenceT, ControlInputPaddingT]
    filter_function: NumPyFilterFunction[ControlInputSequenceT]

    @staticmethod
    def create[
        S: NumPyState,
        SB: NumPyStateBatch,
        CIS: NumPyControlInputSequence,
        CIB: NumPyControlInputBatch,
        CIP: NumPyControlInputSequence = CIS,
    ](
        *,
        planning_interval: int = 1,
        model: NumPyDynamicalModel[S, Any, SB, CIS, CIB],
        cost_function: NumPyCostFunction,
        sampler: NumPySampler[CIS, CIB],
        update_function: NumPyUpdateFunction[CIS] | None = None,
        padding_function: NumPyPaddingFunction[CIS, CIP] | None = None,
        filter_function: NumPyFilterFunction[CIS] | None = None,
    ) -> "NumPyMppi[S, SB, CIS, CIB, CIP]":
        return NumPyMppi(
            planning_interval=planning_interval,
            model=model,
            cost_function=cost_function,
            sampler=sampler,
            update_function=update_function
            or cast(NumPyUpdateFunction[CIS], UseOptimalControlUpdate()),
            padding_function=padding_function
            or cast(NumPyPaddingFunction[CIS, CIP], NumPyZeroPadding()),
            filter_function=filter_function
            or cast(NumPyFilterFunction[CIS], NoFilter()),
        )

    def __post_init__(self) -> None:
        assert self.planning_interval > 0, "Planning interval must be positive."

    def step(
        self,
        *,
        temperature: float,
        nominal_input: ControlInputSequenceT,
        initial_state: StateT,
    ) -> Control[ControlInputSequenceT, NumPyWeights]:
        """Samples rollouts, evaluates costs, and returns the cost-weighted optimal control."""
        assert temperature > 0.0, "Temperature must be positive."

        samples = self.sampler.sample(around=nominal_input)

        rollouts = self.model.simulate(inputs=samples, initial_state=initial_state)
        costs = self.cost_function(inputs=samples, states=rollouts)
        costs_per_rollout = np.sum(costs.array, axis=0)

        min_cost = np.min(costs_per_rollout)
        exp_costs = np.exp((costs_per_rollout - min_cost) / (-temperature))

        normalizing_constant = exp_costs.sum()
        weights = exp_costs / normalizing_constant

        optimal_control = np.tensordot(samples, weights, axes=([2], [0]))
        optimal_input = self.filter_function(
            optimal_input=nominal_input.similar(array=optimal_control)
        )

        nominal_input = self.update_function(
            nominal_input=nominal_input, optimal_input=optimal_input
        )

        shifted_control = np.concat(
            [
                nominal_input.array[self.planning_interval :],
                self.padding_function(
                    nominal_input=nominal_input, padding_size=self.planning_interval
                ).array,
            ],
            axis=0,
        )

        return Control(
            optimal=optimal_input,
            nominal=nominal_input.similar(array=shifted_control),
            debug=DebugData(trajectory_weights=NumPyWeights(weights)),
        )
