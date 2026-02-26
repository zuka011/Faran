from typing import Protocol, Any
from dataclasses import dataclass
from functools import cached_property

from faran.types.array import Array, DataType, jaxtyped
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

from jaxtyping import Float

import numpy as np


class NumPySampledObstacleStates(SampledObstacleStates, Protocol):
    @property
    def array(self) -> Float[Array, "T D_o K N"]:
        """Returns the sampled states of obstacles as a NumPy array."""
        ...


class NumPyObstacleStatesForTimeStep[ObstacleStatesT](
    ObstacleStatesForTimeStep[ObstacleStatesT], Protocol
):
    @property
    def array(self) -> Float[Array, "D_o K"]:
        """Returns the states of obstacles at a specific time step as a NumPy array."""
        ...


class NumPyObstacleStates[
    SingleSampleT,
    ObstacleStatesForTimeStepT = Any,
](ObstacleStates[SingleSampleT], Protocol):
    def last(self) -> ObstacleStatesForTimeStepT:
        """Returns the states of obstacles at the last time step."""
        ...

    @property
    def array(self) -> Float[Array, "T D_o K"]:
        """Returns the states of obstacles as a NumPy array."""
        ...


@jaxtyped
@dataclass(frozen=True)
class NumPySampledObstaclePositions(SampledObstaclePositions):
    _x: Float[Array, "T K N"]
    _y: Float[Array, "T K N"]

    @staticmethod
    def create(
        *, x: Float[Array, "T K N"], y: Float[Array, "T K N"]
    ) -> "NumPySampledObstaclePositions":
        return NumPySampledObstaclePositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 K N"]:
        return self.array

    def x(self) -> Float[Array, "T K N"]:
        return self._x

    def y(self) -> Float[Array, "T K N"]:
        return self._y

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
    def array(self) -> Float[Array, "T 2 K N"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "T 2 K N"]:
        return np.stack([self._x, self._y], axis=1)


@jaxtyped
@dataclass(frozen=True)
class NumPySampledObstacleHeadings(SampledObstacleHeadings):
    _heading: Float[Array, "T K N"]

    @staticmethod
    def create(*, heading: Float[Array, "T K N"]) -> "NumPySampledObstacleHeadings":
        return NumPySampledObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T K N"]:
        return self.array

    def heading(self) -> Float[Array, "T K N"]:
        return self._heading

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
    def array(self) -> Float[Array, "T K N"]:
        return self._heading


class NumPyObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class NumPySampledObstaclePositionExtractor[SampledStatesT](
    SampledObstaclePositionExtractor[SampledStatesT, NumPySampledObstaclePositions],
    Protocol,
): ...


class NumPySampledObstacleHeadingExtractor[SampledStatesT](
    SampledObstacleHeadingExtractor[SampledStatesT, NumPySampledObstacleHeadings],
    Protocol,
): ...


class NumPyDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


@jaxtyped
@dataclass(frozen=True)
class NumPyRisk(Risk):
    _array: Float[Array, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        return self._array

    @property
    def horizon(self) -> int:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._array.shape[1]

    @property
    def array(self) -> Float[Array, "T M"]:
        return self._array


class NumPyRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT](
    RiskMetric, Protocol
):
    def compute(
        self,
        cost_function: SampleCostFunction[
            StateBatchT, SampledObstacleStatesT, Float[Array, "T M N"]
        ],
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT],
    ) -> NumPyRisk:
        """Computes the risk metric based on the provided cost function and returns it as a NumPy array."""
        ...
