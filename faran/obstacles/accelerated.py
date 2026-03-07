from typing import Sequence
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
    DataType,
    POSE_D_O as D_O,
    Device,
    place,
    JaxSampledObstacleStates,
    JaxSampledObstaclePositions,
    JaxSampledObstacleHeadings,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
    JaxObstaclePositions,
    JaxObstaclePositionsForTimeStep,
    JaxObstacleOrientations,
    JaxObstacleOrientationsForTimeStep,
)
from faran.obstacles.basic import NumPyObstacle2dPosesForTimeStep

from numtypes import D
from jaxtyping import Array as JaxArray, Float

import numpy as np
import jax.numpy as jnp

ObstacleCovarianceArray = Float[JaxArray, f"T {D_O} {D_O} K"]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxSampledObstacle2dPoses(JaxSampledObstacleStates):
    """Sampled 2D poses (x, y, heading) with shape (T, POSE_D_O, K, N)."""

    _x: Float[JaxArray, "T K N"]
    _y: Float[JaxArray, "T K N"]
    _heading: Float[JaxArray, "T K N"]

    @staticmethod
    def create(
        *,
        x: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
        y: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
        heading: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
    ) -> "JaxSampledObstacle2dPoses":
        return JaxSampledObstacle2dPoses(
            _x=jnp.asarray(x), _y=jnp.asarray(y), _heading=jnp.asarray(heading)
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K N"]:
        return self._numpy_array

    def x(self) -> Float[Array, "T K N"]:
        return np.asarray(self._x)

    def y(self) -> Float[Array, "T K N"]:
        return np.asarray(self._y)

    def heading(self) -> Float[Array, "T K N"]:
        return np.asarray(self._heading)

    def positions(self) -> JaxSampledObstaclePositions:
        return JaxSampledObstaclePositions.create(x=self._x, y=self._y)

    def headings(self) -> JaxSampledObstacleHeadings:
        return JaxSampledObstacleHeadings.create(heading=self._heading)

    def at(self, *, time_step: int, sample: int) -> "JaxObstacle2dPosesForTimeStep":
        return JaxObstacle2dPosesForTimeStep.create(
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
    def x_array(self) -> Float[JaxArray, "T K N"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        return self._heading

    @property
    def array(self) -> Float[JaxArray, f"T {D_O} K N"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, f"T {D_O} K N"]:
        return jnp.stack([self._x, self._y, self._heading], axis=1)

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_o K N"]:
        return np.asarray(self._array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPositions(JaxObstaclePositions):
    """2D positions (x, y) with shape (T, 2, K)."""

    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]

    @staticmethod
    def create(
        *, x: Float[JaxArray, "T K"], y: Float[JaxArray, "T K"]
    ) -> "JaxObstacle2dPositions":
        return JaxObstacle2dPositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 K"]:
        return self._numpy_array

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
    def array(self) -> Float[JaxArray, "T 2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "T 2 K"]:
        return jnp.stack([self._x, self._y], axis=1)

    @cached_property
    def _numpy_array(self) -> Float[Array, "T 2 K"]:
        return np.asarray(self._array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPositionsForTimeStep(JaxObstaclePositionsForTimeStep):
    """2D positions (x, y) for a single time step with shape (2, K)."""

    _x: Float[JaxArray, " K"]
    _y: Float[JaxArray, " K"]

    @staticmethod
    def create(
        *, x: Float[JaxArray, " K"], y: Float[JaxArray, " K"]
    ) -> "JaxObstacle2dPositionsForTimeStep":
        return JaxObstacle2dPositionsForTimeStep(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "2 K"]:
        return self._numpy_array

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> int:
        return self._x.shape[0]

    @property
    def array(self) -> Float[JaxArray, "2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "2 K"]:
        return jnp.stack([self._x, self._y], axis=0)

    @cached_property
    def _numpy_array(self) -> Float[Array, "2 K"]:
        return np.asarray(self._array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacleHeadings(JaxObstacleOrientations):
    """Obstacle headings with shape (T, 1, K)."""

    _heading: Float[JaxArray, "T K"]

    @staticmethod
    def create(*, heading: Float[JaxArray, "T K"]) -> "JaxObstacleHeadings":
        return JaxObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 1 K"]:
        return self._numpy_array

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
    def array(self) -> Float[JaxArray, "T 1 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "T 1 K"]:
        return self._heading[:, jnp.newaxis, :]

    @cached_property
    def _numpy_array(self) -> Float[Array, "T 1 K"]:
        return np.asarray(self._array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacleHeadingsForTimeStep(JaxObstacleOrientationsForTimeStep):
    """Obstacle headings for a single time step with shape (1, K)."""

    _heading: Float[JaxArray, " K"]

    @staticmethod
    def create(*, heading: Float[JaxArray, " K"]) -> "JaxObstacleHeadingsForTimeStep":
        return JaxObstacleHeadingsForTimeStep(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "1 K"]:
        return self._numpy_array

    @property
    def dimension(self) -> D[1]:
        return 1

    @property
    def count(self) -> int:
        return self._heading.shape[0]

    @property
    def array(self) -> Float[JaxArray, "1 K"]:
        return self._jax_array

    @cached_property
    def _jax_array(self) -> Float[JaxArray, "1 K"]:
        return self._heading[jnp.newaxis, :]

    @cached_property
    def _numpy_array(self) -> Float[Array, "1 K"]:
        return np.asarray(self._jax_array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPoses(
    JaxObstacleStates[
        JaxSampledObstacle2dPoses,
        "JaxObstacle2dPosesForTimeStep",
    ]
):
    """2D poses (x, y, heading) with shape (T, POSE_D_O, K)."""

    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]
    _heading: Float[JaxArray, "T K"]
    _covariance: ObstacleCovarianceArray | None = None

    @staticmethod
    def empty(*, horizon: int, obstacle_count: int = 0) -> "JaxObstacle2dPoses":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = jnp.full((horizon, obstacle_count), fill_value=jnp.nan)

        return JaxObstacle2dPoses.create(x=empty, y=empty, heading=empty)

    @staticmethod
    def sampled(  # type: ignore
        *,
        x: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
        y: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
        heading: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
    ) -> JaxSampledObstacle2dPoses:
        return JaxSampledObstacle2dPoses.create(
            x=jnp.asarray(x), y=jnp.asarray(y), heading=jnp.asarray(heading)
        )

    @staticmethod
    def wrap(array: Float[JaxArray, f"T {D_O} K"]) -> "JaxObstacle2dPoses":
        return JaxObstacle2dPoses.create(
            x=array[:, 0, :], y=array[:, 1, :], heading=array[:, 2, :]
        )

    @staticmethod
    def create(
        *,
        x: Float[Array, "T K"] | Float[JaxArray, "T K"],
        y: Float[Array, "T K"] | Float[JaxArray, "T K"],
        heading: Float[Array, "T K"] | Float[JaxArray, "T K"],
        covariance: Float[Array, "T D_o D_o K"] | ObstacleCovarianceArray | None = None,
    ) -> "JaxObstacle2dPoses":
        return JaxObstacle2dPoses(
            _x=jnp.asarray(x),
            _y=jnp.asarray(y),
            _heading=jnp.asarray(heading),
            _covariance=jnp.asarray(covariance) if covariance is not None else None,
        )

    @staticmethod
    def of_states(
        obstacle_states: Sequence["JaxObstacle2dPosesForTimeStep"],
    ) -> "JaxObstacle2dPoses":
        assert len(obstacle_states) > 0, "Obstacle states sequence must not be empty."

        K = max(states.count for states in obstacle_states)

        def pad(array: Float[JaxArray, " L"]) -> Float[JaxArray, " K"]:
            return jnp.pad(array, (0, K - len(array)), constant_values=jnp.nan)

        x = jnp.stack([pad(states.x_array) for states in obstacle_states])
        y = jnp.stack([pad(states.y_array) for states in obstacle_states])
        heading = jnp.stack([pad(states.heading_array) for states in obstacle_states])

        return JaxObstacle2dPoses.create(x=x, y=y, heading=heading)

    @staticmethod
    def for_time_step(
        *,
        x: Float[Array, " K"] | Float[JaxArray, " K"],
        y: Float[Array, " K"] | Float[JaxArray, " K"],
        heading: Float[Array, " K"] | Float[JaxArray, " K"],
        device: Device = "cpu",
    ) -> "JaxObstacle2dPosesForTimeStep":
        """Creates obstacle states for a single time step.

        Note:
            Since the common case is to further process this data on the CPU first,
            the default device is set to "cpu".
        """
        return JaxObstacle2dPosesForTimeStep.create(
            x=place(x, device=device),
            y=place(y, device=device),
            heading=place(heading, device=device),
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K"]:
        return self._numpy_array

    def x(self) -> Float[Array, "T K"]:
        return np.asarray(self._x)

    def y(self) -> Float[Array, "T K"]:
        return np.asarray(self._y)

    def heading(self) -> Float[Array, "T K"]:
        return np.asarray(self._heading)

    def covariance(self) -> Float[Array, "T D_o D_o K"] | None:
        return np.asarray(self._covariance) if self._covariance is not None else None

    def positions(self) -> JaxObstacle2dPositions:
        return JaxObstacle2dPositions.create(x=self._x, y=self._y)

    def headings(self) -> JaxObstacleHeadings:
        return JaxObstacleHeadings.create(heading=self._heading)

    def single(self) -> JaxSampledObstacle2dPoses:
        return JaxSampledObstacle2dPoses.create(
            x=self._x[..., jnp.newaxis],
            y=self._y[..., jnp.newaxis],
            heading=self._heading[..., jnp.newaxis],
        )

    def last(self) -> "JaxObstacle2dPosesForTimeStep":
        return self.at(time_step=self.horizon - 1)

    def at(self, time_step: int) -> "JaxObstacle2dPosesForTimeStep":
        return JaxObstacle2dPosesForTimeStep.create(
            x=self._x[time_step],
            y=self._y[time_step],
            heading=self._heading[time_step],
        )

    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        return self._array

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return self._heading

    @property
    def covariance_array(self) -> ObstacleCovarianceArray | None:
        return self._covariance

    @property
    def horizon(self) -> int:
        return self._x.shape[0]

    @property
    def dimension(self) -> int:
        return D_O

    @property
    def count(self) -> int:
        return self._x.shape[1]

    @cached_property
    def _array(self) -> Float[JaxArray, "T D_o K"]:
        return jnp.stack([self._x, self._y, self._heading], axis=1)

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_o K"]:
        return np.asarray(self._array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPosesForTimeStep(JaxObstacleStatesForTimeStep[JaxObstacle2dPoses]):
    """2D poses (x, y, heading) for a single time step."""

    _x: Float[JaxArray, " K"]
    _y: Float[JaxArray, " K"]
    _heading: Float[JaxArray, " K"]

    @staticmethod
    def wrap(
        array: Float[Array, f"{D_O} K"] | Float[JaxArray, f"{D_O} K"],
    ) -> "JaxObstacle2dPosesForTimeStep":
        return JaxObstacle2dPosesForTimeStep.create(
            x=array[0, :], y=array[1, :], heading=array[2, :]
        )

    @staticmethod
    def create(
        *,
        x: Float[Array, " K"] | Float[JaxArray, " K"],
        y: Float[Array, " K"] | Float[JaxArray, " K"],
        heading: Float[Array, " K"] | Float[JaxArray, " K"],
    ) -> "JaxObstacle2dPosesForTimeStep":

        return JaxObstacle2dPosesForTimeStep(
            _x=jnp.asarray(x), _y=jnp.asarray(y), _heading=jnp.asarray(heading)
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "D_o K"]:
        return self._numpy_array

    def numpy(self) -> NumPyObstacle2dPosesForTimeStep:
        return NumPyObstacle2dPosesForTimeStep.create(
            x=np.asarray(self._x),
            y=np.asarray(self._y),
            heading=np.asarray(self._heading),
        )

    def x(self) -> Float[Array, " K"]:
        return np.asarray(self._x)

    def y(self) -> Float[Array, " K"]:
        return np.asarray(self._y)

    def heading(self) -> Float[Array, " K"]:
        return np.asarray(self._heading)

    def positions(self) -> JaxObstacle2dPositionsForTimeStep:
        return JaxObstacle2dPositionsForTimeStep.create(x=self._x, y=self._y)

    def headings(self) -> JaxObstacleHeadingsForTimeStep:
        return JaxObstacleHeadingsForTimeStep.create(heading=self._heading)

    def replicate(self, *, horizon: int) -> JaxObstacle2dPoses:
        return JaxObstacle2dPoses.create(
            x=jnp.tile(self._x[jnp.newaxis, :], (horizon, 1)),
            y=jnp.tile(self._y[jnp.newaxis, :], (horizon, 1)),
            heading=jnp.tile(self._heading[jnp.newaxis, :], (horizon, 1)),
        )

    @property
    def dimension(self) -> int:
        return D_O

    @property
    def count(self) -> int:
        return self._x.shape[0]

    @property
    def x_array(self) -> Float[JaxArray, " K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, " K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, " K"]:
        return self._heading

    @property
    def array(self) -> Float[JaxArray, f"{D_O} K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, f"{D_O} K"]:
        return jnp.stack([self._x, self._y, self._heading], axis=0)

    @cached_property
    def _numpy_array(self) -> Float[Array, "D_o K"]:
        return np.asarray(self._array)
