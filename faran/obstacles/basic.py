from typing import Sequence
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
    DataType,
    POSE_D_O as D_O,
    NumPySampledObstacleStates,
    NumPySampledObstaclePositions,
    NumPySampledObstacleHeadings,
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
    NumPyObstaclePositions,
    NumPyObstaclePositionsForTimeStep,
    NumPyObstacleOrientations,
    NumPyObstacleOrientationsForTimeStep,
)

from numtypes import D
from jaxtyping import Float

import numpy as np

ObstacleCovarianceArray = Float[Array, "T D_o D_o K"]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPySampledObstacle2dPoses(NumPySampledObstacleStates):
    """Sampled 2D poses (x, y, heading) with shape (T, POSE_D_O, K, N)."""

    _x: Float[Array, "T K N"]
    _y: Float[Array, "T K N"]
    _heading: Float[Array, "T K N"]

    @staticmethod
    def create(
        *,
        x: Float[Array, "T K N"],
        y: Float[Array, "T K N"],
        heading: Float[Array, "T K N"],
    ) -> "NumPySampledObstacle2dPoses":
        return NumPySampledObstacle2dPoses(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K N"]:
        return self.array

    def x(self) -> Float[Array, "T K N"]:
        return self._x

    def y(self) -> Float[Array, "T K N"]:
        return self._y

    def heading(self) -> Float[Array, "T K N"]:
        return self._heading

    def positions(self) -> NumPySampledObstaclePositions:
        return NumPySampledObstaclePositions.create(x=self._x, y=self._y)

    def headings(self) -> NumPySampledObstacleHeadings:
        return NumPySampledObstacleHeadings.create(heading=self._heading)

    def at(self, *, time_step: int, sample: int) -> "NumPyObstacle2dPosesForTimeStep":
        return NumPyObstacle2dPosesForTimeStep.create(
            x=self._x[time_step, :, sample],
            y=self._y[time_step, :, sample],
            heading=self._heading[time_step, :, sample],
        )

    @property
    def horizon(self) -> int:
        return self._x.shape[0]

    @property
    def dimension(self) -> int:
        return D_O

    @property
    def count(self) -> int:
        return self._x.shape[1]

    @property
    def sample_count(self) -> int:
        return self._x.shape[2]

    @property
    def array(self) -> Float[Array, "T D_o K N"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "T D_o K N"]:
        return np.stack([self._x, self._y, self._heading], axis=1)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyObstacle2dPositions(NumPyObstaclePositions):
    """2D positions (x, y) with shape (T, 2, K)."""

    _x: Float[Array, "T K"]
    _y: Float[Array, "T K"]

    @staticmethod
    def create(
        *, x: Float[Array, "T K"], y: Float[Array, "T K"]
    ) -> "NumPyObstacle2dPositions":
        return NumPyObstacle2dPositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 K"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self._x.shape[0]

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> int:
        return self._x.shape[1]

    @property
    def array(self) -> Float[Array, "T 2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "T 2 K"]:
        return np.stack([self._x, self._y], axis=1)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyObstacle2dPositionsForTimeStep(NumPyObstaclePositionsForTimeStep):
    """2D positions (x, y) for a single time step with shape (2, K)."""

    _x: Float[Array, " K"]
    _y: Float[Array, " K"]

    @staticmethod
    def create(
        *, x: Float[Array, " K"], y: Float[Array, " K"]
    ) -> "NumPyObstacle2dPositionsForTimeStep":
        return NumPyObstacle2dPositionsForTimeStep(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "2 K"]:
        return self.array

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> int:
        return self._x.shape[0]

    @property
    def array(self) -> Float[Array, "2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "2 K"]:
        return np.stack([self._x, self._y], axis=0)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyObstacleHeadings(NumPyObstacleOrientations):
    """Obstacle headings with shape (T, 1, K)."""

    _heading: Float[Array, "T K"]

    @staticmethod
    def create(*, heading: Float[Array, "T K"]) -> "NumPyObstacleHeadings":
        return NumPyObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 1 K"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self._heading.shape[0]

    @property
    def dimension(self) -> D[1]:
        return 1

    @property
    def count(self) -> int:
        return self._heading.shape[1]

    @property
    def array(self) -> Float[Array, "T 1 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "T 1 K"]:
        return self._heading[:, np.newaxis, :]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyObstacleHeadingsForTimeStep(NumPyObstacleOrientationsForTimeStep):
    """Obstacle headings for a single time step."""

    _heading: Float[Array, " K"]

    @staticmethod
    def create(*, heading: Float[Array, " K"]) -> "NumPyObstacleHeadingsForTimeStep":
        return NumPyObstacleHeadingsForTimeStep(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "1 K"]:
        return self.array

    @property
    def dimension(self) -> D[1]:
        return 1

    @property
    def count(self) -> int:
        return self._heading.shape[0]

    @property
    def array(self) -> Float[Array, "1 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "1 K"]:
        return self._heading[np.newaxis, :]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyObstacle2dPoses(
    NumPyObstacleStates[
        NumPySampledObstacle2dPoses,
        "NumPyObstacle2dPosesForTimeStep",
    ]
):
    """2D poses (x, y, heading) with shape (T, POSE_D_O, K)."""

    _x: Float[Array, "T K"]
    _y: Float[Array, "T K"]
    _heading: Float[Array, "T K"]
    _covariance: ObstacleCovarianceArray | None

    @staticmethod
    def empty(*, horizon: int, obstacle_count: int = 0) -> "NumPyObstacle2dPoses":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = np.full((horizon, obstacle_count), fill_value=np.nan)

        return NumPyObstacle2dPoses.create(x=empty, y=empty, heading=empty)

    @staticmethod
    def wrap(array: Float[Array, "T D_o K"]) -> "NumPyObstacle2dPoses":
        return NumPyObstacle2dPoses.create(
            x=array[:, 0, :], y=array[:, 1, :], heading=array[:, 2, :]
        )

    @staticmethod
    def sampled(  # type: ignore
        *,
        x: Float[Array, "T K N"],
        y: Float[Array, "T K N"],
        heading: Float[Array, "T K N"],
    ) -> NumPySampledObstacle2dPoses:
        return NumPySampledObstacle2dPoses.create(x=x, y=y, heading=heading)

    @staticmethod
    def create(
        *,
        x: Float[Array, "T K"],
        y: Float[Array, "T K"],
        heading: Float[Array, "T K"],
        covariance: ObstacleCovarianceArray | None = None,
    ) -> "NumPyObstacle2dPoses":
        return NumPyObstacle2dPoses(
            _x=x, _y=y, _heading=heading, _covariance=covariance
        )

    @staticmethod
    def of_states(
        obstacle_states: Sequence["NumPyObstacle2dPosesForTimeStep"],
    ) -> "NumPyObstacle2dPoses":
        assert len(obstacle_states) > 0, "Obstacle states sequence must not be empty."

        T = len(obstacle_states)
        K = max(states.count for states in obstacle_states)

        x = np.full((T, K), np.nan)
        y = np.full((T, K), np.nan)
        heading = np.full((T, K), np.nan)

        for t, states in enumerate(obstacle_states):
            n = states.count
            x[t, :n] = states.x()
            y[t, :n] = states.y()
            heading[t, :n] = states.heading()

        return NumPyObstacle2dPoses.create(x=x, y=y, heading=heading)

    @staticmethod
    def for_time_step(
        *, x: Float[Array, " K"], y: Float[Array, " K"], heading: Float[Array, " K"]
    ) -> "NumPyObstacle2dPosesForTimeStep":
        return NumPyObstacle2dPosesForTimeStep.create(x=x, y=y, heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K"]:
        return self.array

    def x(self) -> Float[Array, "T K"]:
        return self._x

    def y(self) -> Float[Array, "T K"]:
        return self._y

    def heading(self) -> Float[Array, "T K"]:
        return self._heading

    def positions(self) -> NumPyObstacle2dPositions:
        return NumPyObstacle2dPositions.create(x=self._x, y=self._y)

    def headings(self) -> NumPyObstacleHeadings:
        return NumPyObstacleHeadings.create(heading=self._heading)

    def covariance(self) -> ObstacleCovarianceArray | None:
        return self._covariance

    def single(self) -> NumPySampledObstacle2dPoses:
        return NumPySampledObstacle2dPoses.create(
            x=self._x[..., np.newaxis],
            y=self._y[..., np.newaxis],
            heading=self._heading[..., np.newaxis],
        )

    def last(self) -> "NumPyObstacle2dPosesForTimeStep":
        return self.at(time_step=self.horizon - 1)

    def at(self, *, time_step: int) -> "NumPyObstacle2dPosesForTimeStep":
        return NumPyObstacle2dPosesForTimeStep.create(
            x=self._x[time_step],
            y=self._y[time_step],
            heading=self._heading[time_step],
        )

    @property
    def horizon(self) -> int:
        return self._x.shape[0]

    @property
    def dimension(self) -> int:
        return D_O

    @property
    def count(self) -> int:
        return self._x.shape[1]

    @property
    def array(self) -> Float[Array, "T D_o K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "T D_o K"]:
        return np.stack([self._x, self._y, self._heading], axis=1)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class NumPyObstacle2dPosesForTimeStep(
    NumPyObstacleStatesForTimeStep[NumPyObstacle2dPoses]
):
    """2D poses (x, y, heading) for a single time step with shape (POSE_D_O, K)."""

    _x: Float[Array, " K"]
    _y: Float[Array, " K"]
    _heading: Float[Array, " K"]

    @staticmethod
    def create(
        *,
        x: Float[Array, " K"],
        y: Float[Array, " K"],
        heading: Float[Array, " K"],
    ) -> "NumPyObstacle2dPosesForTimeStep":
        return NumPyObstacle2dPosesForTimeStep(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "D_o K"]:
        return self.array

    def x(self) -> Float[Array, " K"]:
        return self._x

    def y(self) -> Float[Array, " K"]:
        return self._y

    def heading(self) -> Float[Array, " K"]:
        return self._heading

    def positions(self) -> NumPyObstacle2dPositionsForTimeStep:
        return NumPyObstacle2dPositionsForTimeStep.create(x=self._x, y=self._y)

    def headings(self) -> NumPyObstacleHeadingsForTimeStep:
        return NumPyObstacleHeadingsForTimeStep.create(heading=self._heading)

    def replicate(self, *, horizon: int) -> NumPyObstacle2dPoses:
        return NumPyObstacle2dPoses.create(
            x=np.tile(self._x[np.newaxis, :], (horizon, 1)),
            y=np.tile(self._y[np.newaxis, :], (horizon, 1)),
            heading=np.tile(self._heading[np.newaxis, :], (horizon, 1)),
        )

    @property
    def dimension(self) -> int:
        return D_O

    @property
    def count(self) -> int:
        return self._x.shape[0]

    @property
    def array(self) -> Float[Array, "D_o K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "D_o K"]:
        return np.stack([self._x, self._y, self._heading], axis=0)
