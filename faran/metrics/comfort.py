from typing import Any
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    Array,
    StateSequence,
    SimulationData,
    Metric,
    Positions,
    PositionExtractor,
    Trajectory,
    LateralPositions,
)
from faran.collectors import access

from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class ComfortMetricResult:
    """Results of the comfort metric, including lateral acceleration and jerk."""

    lateral_acceleration: Float[Array, " T"]
    lateral_jerk: Float[Array, " T"]


@dataclass(kw_only=True, frozen=True)
class ComfortMetric[StateBatchT, PositionsT, LateralT](Metric[ComfortMetricResult]):
    """Metric evaluating lateral acceleration and jerk relative to a reference trajectory."""

    reference: Trajectory[Any, Any, PositionsT, LateralT]
    time_step_size: float
    position_extractor: PositionExtractor[StateBatchT, PositionsT]

    @staticmethod
    def create[S, P: Positions, L: LateralPositions](
        *,
        reference: Trajectory[Any, Any, P, L],
        time_step_size: float,
        position_extractor: PositionExtractor[S, P],
    ) -> "ComfortMetric[S, P, L]":
        assert time_step_size > 0, (
            f"Time step size must be positive, got {time_step_size}"
        )

        return ComfortMetric(
            reference=reference,
            time_step_size=time_step_size,
            position_extractor=position_extractor,
        )

    def compute(self, data: SimulationData) -> ComfortMetricResult:
        states = data(access.states.assume(StateSequence[StateBatchT]).require())
        state_batch = states.batched()

        positions = self.position_extractor(state_batch)
        lateral = self.reference.lateral(positions)

        lateral_array = np.asarray(lateral)[:, 0]
        lateral_velocity = checked_gradient(lateral_array, self.time_step_size)
        lateral_acceleration = checked_gradient(lateral_velocity, self.time_step_size)
        lateral_jerk = checked_gradient(lateral_acceleration, self.time_step_size)

        return ComfortMetricResult(
            lateral_acceleration=lateral_acceleration, lateral_jerk=lateral_jerk
        )

    @property
    def name(self) -> str:
        return "comfort"


def checked_gradient(y: Float[Array, " L"], dx: float) -> Float[Array, " L"]:
    if y.shape[0] < 2:
        return np.zeros_like(y)

    return np.gradient(y, dx)
