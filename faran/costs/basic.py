from typing import Any
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    Array,
    DataType,
    StateBatch,
    ControlInputBatch,
    CostFunction,
    ContouringCost,
    LagCost,
    Error,
    Trajectory,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositionExtractor,
    NumPyPathParameterExtractor,
    NumPyPathVelocityExtractor,
)
from faran.states import NumPySimpleCosts

from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(frozen=True)
class NumPyError(Error):
    """Contouring or lag error between the state batch and reference trajectory."""

    array: Float[Array, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        return self.array


@dataclass(kw_only=True, frozen=True)
class NumPyContouringCost[StateBatchT](
    ContouringCost[ControlInputBatch, StateBatchT, NumPyError],
    CostFunction[ControlInputBatch, StateBatchT, NumPyCosts],
):
    """Penalizes lateral deviation from a reference trajectory."""

    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints]
    path_parameter_extractor: NumPyPathParameterExtractor[StateBatchT]
    position_extractor: NumPyPositionExtractor[StateBatchT]
    weight: float

    @staticmethod
    def create[S](
        *,
        reference: Trajectory[NumPyPathParameters, NumPyReferencePoints],
        path_parameter_extractor: NumPyPathParameterExtractor[S],
        position_extractor: NumPyPositionExtractor[S],
        weight: float,
    ) -> "NumPyContouringCost[S]":
        """Creates a contouring cost implemented with NumPy.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the contouring cost.
        """
        return NumPyContouringCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__(self, *, inputs: ControlInputBatch, states: StateBatchT) -> NumPyCosts:
        error = self.error(states=states)
        return NumPySimpleCosts(self.weight * error.array**2)

    def error(self, *, states: StateBatchT) -> NumPyError:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading()
        positions = self.position_extractor(states)

        return NumPyError(
            np.sin(heading) * (positions.x() - ref_points.x())
            - np.cos(heading) * (positions.y() - ref_points.y())
        )


@dataclass(kw_only=True, frozen=True)
class NumPyLagCost[StateBatchT](
    LagCost[ControlInputBatch, StateBatchT, NumPyError],
    CostFunction[ControlInputBatch, StateBatchT, NumPyCosts],
):
    """Penalizes longitudinal deviation from a reference trajectory."""

    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints]
    path_parameter_extractor: NumPyPathParameterExtractor[StateBatchT]
    position_extractor: NumPyPositionExtractor[StateBatchT]
    weight: float

    @staticmethod
    def create[S](
        *,
        reference: Trajectory[NumPyPathParameters, NumPyReferencePoints],
        path_parameter_extractor: NumPyPathParameterExtractor[S],
        position_extractor: NumPyPositionExtractor[S],
        weight: float,
    ) -> "NumPyLagCost[S]":
        """Creates a lag cost implemented with NumPy.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the lag cost.
        """
        return NumPyLagCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__(self, *, inputs: ControlInputBatch, states: StateBatchT) -> NumPyCosts:
        error = self.error(states=states)
        return NumPySimpleCosts(self.weight * error.array**2)

    def error(self, *, states: StateBatchT) -> NumPyError:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading()
        positions = self.position_extractor(states)

        return NumPyError(
            -np.cos(heading) * (positions.x() - ref_points.x())
            - np.sin(heading) * (positions.y() - ref_points.y())
        )


@dataclass(kw_only=True, frozen=True)
class NumPyProgressCost[InputBatchT](CostFunction[InputBatchT, StateBatch, NumPyCosts]):
    """Rewards forward progress along a reference trajectory."""

    path_velocity_extractor: NumPyPathVelocityExtractor[InputBatchT]
    time_step_size: float
    weight: float

    @staticmethod
    def create[I](
        *,
        path_velocity_extractor: NumPyPathVelocityExtractor[I],
        time_step_size: float,
        weight: float,
    ) -> "NumPyProgressCost[I]":
        """Creates a progress cost implemented with NumPy.

        Args:
            path_velocity_extractor: Extracts the path velocities from a control input batch.
            time_step_size: The time step size between states.
            weight: The weight of the progress cost.
        """
        return NumPyProgressCost(
            path_velocity_extractor=path_velocity_extractor,
            time_step_size=time_step_size,
            weight=weight,
        )

    def __call__(self, *, inputs: InputBatchT, states: StateBatch) -> NumPyCosts:
        path_velocities = self.path_velocity_extractor(inputs)

        return NumPySimpleCosts(-self.weight * path_velocities * self.time_step_size)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyControlSmoothingCost(
    CostFunction[NumPyControlInputBatch, StateBatch, NumPyCosts]
):
    """Penalizes abrupt changes in control inputs between consecutive time steps."""

    weights: Float[Array, " D_u"]

    @staticmethod
    def create(
        *,
        weights: Float[Array, " D_u"],
    ) -> "NumPyControlSmoothingCost":
        """Creates a control smoothing cost implemented with NumPy.

        Args:
            weights: The weights for each control input dimension.
        """
        return NumPyControlSmoothingCost(weights=weights)

    def __call__(
        self, *, inputs: NumPyControlInputBatch, states: StateBatch
    ) -> NumPyCosts:
        diffs = np.diff(inputs.array, axis=0, prepend=inputs.array[0:1, :, :])
        squared_diffs = diffs**2
        weighted_squared_diffs = squared_diffs * self.weights[np.newaxis, :, np.newaxis]
        cost_per_time_step = np.sum(weighted_squared_diffs, axis=1)

        return NumPySimpleCosts(cost_per_time_step)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyControlEffortCost(CostFunction[NumPyControlInputBatch, Any, NumPyCosts]):
    """Penalizes large control input magnitudes."""

    weights: Float[Array, " D_u"]

    @staticmethod
    def create(*, weights: Float[Array, " D_u"]) -> "NumPyControlEffortCost":
        """Creates a control effort cost implemented with NumPy.

        Args:
            weights: The weights for each control input dimension.
        """
        return NumPyControlEffortCost(weights=weights)

    def __call__(self, *, inputs: NumPyControlInputBatch, states: Any) -> NumPyCosts:
        return NumPySimpleCosts(np.einsum("u,tum->tm", self.weights, inputs.array**2))
