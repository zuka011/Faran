from typing import Protocol, Any
from dataclasses import dataclass
from functools import cached_property

from faran.types.array import Array, jaxtyped, DataType
from faran.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    SampledObstacleStates,
    SampledObstaclePositions,
    SampledObstacleHeadings,
    SampledObstaclePositionExtractor,
    SampledObstacleHeadingExtractor,
    ObstacleStatesForTimeStep,
    ObstacleStates,
    DistanceExtractor,
    SampleCostFunction,
    Risk,
    RiskMetric,
)

from jaxtyping import Float, Array as JaxArray

import numpy as np


class JaxSampledObstacleStates(SampledObstacleStates, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_o K N"]:
        """Returns the sampled states of obstacles as a JAX array."""
        ...


class JaxObstacleStatesForTimeStep[ObstacleStatesT, NumPyT = Any](
    ObstacleStatesForTimeStep[ObstacleStatesT], Protocol
):
    def numpy(self) -> NumPyT:
        """Returns the states of obstacles at a specific time step wrapped for the NumPy backend."""
        ...

    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        """Returns the states of obstacles at a specific time step as a JAX array."""
        ...


class JaxObstacleStates[SingleSampleT, ObstacleStatesForTimeStepT = Any](
    ObstacleStates[SingleSampleT], Protocol
):
    def last(self) -> ObstacleStatesForTimeStepT:
        """Returns the states of obstacles at the last time step."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the states of obstacles as a JAX array."""
        ...

    @property
    def covariance_array(self) -> Float[JaxArray, "T D_o D_o K"] | None:
        """Returns the covariances of obstacles as a JAX array, or None if not available."""
        ...


@jaxtyped
@dataclass(frozen=True)
class JaxSampledObstaclePositions(SampledObstaclePositions):
    _x: Float[JaxArray, "T K N"]
    _y: Float[JaxArray, "T K N"]

    @staticmethod
    def create(
        *, x: Float[JaxArray, "T K N"], y: Float[JaxArray, "T K N"]
    ) -> "JaxSampledObstaclePositions":
        return JaxSampledObstaclePositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 K N"]:
        return self._numpy_array

    def x(self) -> Float[Array, "T K N"]:
        return self._numpy_x

    def y(self) -> Float[Array, "T K N"]:
        return self._numpy_y

    @property
    def horizon(self) -> int:
        return self._x.shape[0]

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

    @cached_property
    def _numpy_x(self) -> Float[Array, "T K N"]:
        return np.asarray(self._x)

    @cached_property
    def _numpy_y(self) -> Float[Array, "T K N"]:
        return np.asarray(self._y)

    @cached_property
    def _numpy_array(self) -> Float[Array, "T 2 K N"]:
        return np.stack([self._numpy_x, self._numpy_y], axis=1)


@jaxtyped
@dataclass(frozen=True)
class JaxSampledObstacleHeadings(SampledObstacleHeadings):
    _heading: Float[JaxArray, "T K N"]

    @staticmethod
    def create(*, heading: Float[JaxArray, "T K N"]) -> "JaxSampledObstacleHeadings":
        return JaxSampledObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T K N"]:
        return self._numpy_heading

    def heading(self) -> Float[Array, "T K N"]:
        return self._numpy_heading

    @property
    def horizon(self) -> int:
        return self._heading.shape[0]

    @property
    def count(self) -> int:
        return self._heading.shape[1]

    @property
    def sample_count(self) -> int:
        return self._heading.shape[2]

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        return self._heading

    @cached_property
    def _numpy_heading(self) -> Float[Array, "T K N"]:
        return np.asarray(self._heading)


class JaxObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class JaxSampledObstaclePositionExtractor[SampledStatesT](
    SampledObstaclePositionExtractor[SampledStatesT, JaxSampledObstaclePositions],
    Protocol,
): ...


class JaxSampledObstacleHeadingExtractor[SampledStatesT](
    SampledObstacleHeadingExtractor[SampledStatesT, JaxSampledObstacleHeadings],
    Protocol,
): ...


class JaxDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


@jaxtyped
@dataclass(frozen=True)
class JaxRisk(Risk):
    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        return np.asarray(self.array)

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array


class JaxRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT](
    RiskMetric, Protocol
):
    def compute(
        self,
        cost_function: SampleCostFunction[
            StateBatchT, SampledObstacleStatesT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT],
    ) -> JaxRisk:
        """Computes the risk metric based on the provided cost function and returns it as a JAX array."""
        ...
