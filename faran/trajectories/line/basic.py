from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
    Trajectory,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyLateralPositions,
    NumPyLongitudinalPositions,
    NumPyNormals,
)

from jaxtyping import Float

import numpy as np


type Vector = Float[Array, "2"]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyLineTrajectory(
    Trajectory[
        NumPyPathParameters,
        NumPyReferencePoints,
        NumPyPositions,
        NumPyLateralPositions,
        NumPyLongitudinalPositions,
    ]
):
    """NumPy straight-line reference trajectory from start to end."""

    start: Vector
    direction: Vector

    heading: float

    _end: tuple[float, float]
    _path_length: float

    @staticmethod
    def create(
        *, start: tuple[float, float], end: tuple[float, float], path_length: float
    ) -> "NumPyLineTrajectory":
        """Generates a straight line trajectory from start to end."""
        return NumPyLineTrajectory(
            start=(start_array := np.array(start)),
            direction=(direction := np.array(end) - start_array),
            heading=np.arctan2(direction[1], direction[0]),
            _end=end,
            _path_length=path_length,
        )

    def query(self, parameters: NumPyPathParameters) -> NumPyReferencePoints:
        normalized = parameters.array / self.path_length

        x, y = (
            self.start[:, np.newaxis, np.newaxis]
            + self.direction[:, np.newaxis, np.newaxis] * normalized
        )
        heading = np.full_like(x, self.heading)

        return NumPyReferencePoints.create(x=x, y=y, heading=heading)

    def lateral(self, positions: NumPyPositions) -> NumPyLateralPositions:
        relative = positions.array - self.start[:, np.newaxis]
        lateral = np.einsum("tpm,p->tm", relative, self.perpendicular)

        return NumPyLateralPositions.create(lateral)

    def longitudinal(self, positions: NumPyPositions) -> NumPyLongitudinalPositions:
        relative = positions.array - self.start[:, np.newaxis]
        projection = np.einsum("tpm,p->tm", relative, self.tangent)
        longitudinal = projection * self.path_length / self.line_length

        return NumPyLongitudinalPositions.create(longitudinal)

    def normal(self, parameters: NumPyPathParameters) -> NumPyNormals:
        T, M = parameters.horizon, parameters.rollout_count

        x = np.full((T, M), self.perpendicular[0])
        y = np.full((T, M), self.perpendicular[1])

        return NumPyNormals.create(x=x, y=y)

    @property
    def end(self) -> tuple[float, float]:
        return self._end

    @property
    def path_length(self) -> float:
        return self._path_length

    @property
    def natural_length(self) -> float:
        return self.line_length

    @cached_property
    def perpendicular(self) -> Vector:
        tangent = self.tangent
        perpendicular = np.array([tangent[1], -tangent[0]])

        return perpendicular

    @cached_property
    def tangent(self) -> Vector:
        tangent = self.direction / self.line_length

        return tangent

    @cached_property
    def line_length(self) -> float:
        return float(np.linalg.norm(self.direction))
