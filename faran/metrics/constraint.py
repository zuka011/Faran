from typing import Any
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
    StateSequence,
    SimulationData,
    Metric,
    Positions,
    LateralPositions,
    PositionExtractor,
    Trajectory,
    BoundaryDistance,
    BoundaryDistanceExtractor,
)
from faran.collectors import access

from jaxtyping import Float, Bool

import numpy as np


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class ConstraintViolationMetricResult:
    """Results of the constraint violation metric, including boundary distances."""

    lateral_deviations: Float[Array, " T"]
    boundary_distances: Float[Array, " T"]

    @cached_property
    def violations(self) -> Bool[Array, " T"]:
        return self.boundary_distances <= 0

    @cached_property
    def violation_detected(self) -> bool:
        return bool(self.violations.any())


@dataclass(kw_only=True, frozen=True)
class ConstraintViolationMetric[StateBatchT, PositionsT, LateralT, BoundaryDistanceT](
    Metric[ConstraintViolationMetricResult]
):
    """Metric evaluating lateral deviations and boundary constraint violations."""

    reference: Trajectory[Any, Any, PositionsT, LateralT]
    boundary: BoundaryDistanceExtractor[StateBatchT, BoundaryDistanceT]
    position_extractor: PositionExtractor[StateBatchT, PositionsT]

    @staticmethod
    def create[S, P: Positions, L: LateralPositions, BD: BoundaryDistance](
        *,
        reference: Trajectory[Any, Any, P, L],
        boundary: BoundaryDistanceExtractor[S, BD],
        position_extractor: PositionExtractor[S, P],
    ) -> "ConstraintViolationMetric[S, P, L, BD]":
        return ConstraintViolationMetric(
            reference=reference,
            boundary=boundary,
            position_extractor=position_extractor,
        )

    def compute(self, data: SimulationData) -> ConstraintViolationMetricResult:
        states = data(access.states.assume(StateSequence[StateBatchT]).require())
        state_batch = states.batched()

        positions = self.position_extractor(state_batch)
        lateral = self.reference.lateral(positions)
        boundary_distance = self.boundary(states=state_batch)

        lateral_array = np.asarray(lateral)[:, 0]
        boundary_array = np.asarray(boundary_distance)[:, 0]

        return ConstraintViolationMetricResult(
            lateral_deviations=lateral_array, boundary_distances=boundary_array
        )

    @property
    def name(self) -> str:
        return "constraint-violation"
