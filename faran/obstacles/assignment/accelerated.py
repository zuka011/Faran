from dataclasses import dataclass

from faran.types import (
    DataType,
    ObstacleIdAssignment,
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    JaxObstaclePositionsForTimeStep,
    JaxObstaclePositions,
    JaxObstaclePositionExtractor,
    JaxObstacleOrientationsForTimeStep,
    JaxObstacleOrientations,
    JaxObstacleOrientationExtractor,
    NumPyObstaclePositionsForTimeStep,
    NumPyObstaclePositions,
    NumPyObstaclePositionExtractor,
    NumPyObstacleOrientationsForTimeStep,
    NumPyObstacleOrientations,
    NumPyObstacleOrientationExtractor,
)
from faran.obstacles.history.accelerated import JaxObstacleIds
from faran.obstacles.history.basic import NumPyObstacleIds
from faran.obstacles.assignment.basic import NumPyHungarianObstacleIdAssignment

from numtypes import Array, Dims

import numpy as np
import jax.numpy as jnp


type JaxPositionsExtractor[StatesT, HistoryT] = JaxObstaclePositionExtractor[
    StatesT, HistoryT, JaxObstaclePositionsForTimeStep, JaxObstaclePositions
]

type JaxOrientationsExtractor[StatesT, HistoryT] = JaxObstacleOrientationExtractor[
    StatesT,
    HistoryT,
    JaxObstacleOrientationsForTimeStep,
    JaxObstacleOrientations,
]


@dataclass(frozen=True)
class NumPyAdaptedObstaclePositionsForTimeStep[D_p: int, K: int](
    NumPyObstaclePositionsForTimeStep[D_p, K]
):
    """Adapts JAX obstacle positions to NumPy for a single time step."""

    _array: Array[Dims[D_p, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_p, K]]:
        return self.array

    @property
    def dimension(self) -> D_p:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_p, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyAdaptedObstaclePositions[T: int, D_p: int, K: int](
    NumPyObstaclePositions[T, D_p, K]
):
    """Adapts JAX obstacle positions to NumPy across time steps."""

    _array: Array[Dims[T, D_p, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_p, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_p:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_p, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyAdaptedObstacleOrientationsForTimeStep[D_o: int, K: int](
    NumPyObstacleOrientationsForTimeStep[D_o, K]
):
    """Adapts JAX obstacle orientations to NumPy for a single time step."""

    _array: Array[Dims[D_o, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return self.array

    @property
    def dimension(self) -> D_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyAdaptedObstacleOrientations[T: int, D_o: int, K: int](
    NumPyObstacleOrientations[T, D_o, K]
):
    """Adapts JAX obstacle orientations to NumPy across time steps."""

    _array: Array[Dims[T, D_o, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        return self._array


@dataclass(kw_only=True)
class PositionExtractorAdapter[StatesT, HistoryT](
    NumPyObstaclePositionExtractor[
        StatesT, HistoryT, NumPyObstaclePositionsForTimeStep, NumPyObstaclePositions
    ]
):
    """Adapts a JAX position extractor to the NumPy interface for obstacle assignment."""

    inner: JaxObstaclePositionExtractor[
        StatesT, HistoryT, JaxObstaclePositionsForTimeStep, JaxObstaclePositions
    ]

    @staticmethod
    def adapt(
        extractor: JaxPositionsExtractor[StatesT, HistoryT],
    ) -> "PositionExtractorAdapter[StatesT, HistoryT]":
        return PositionExtractorAdapter(inner=extractor)

    def of_states_for_time_step(
        self, states: StatesT, /
    ) -> NumPyObstaclePositionsForTimeStep:
        return NumPyAdaptedObstaclePositionsForTimeStep(
            np.asarray(self.inner.of_states_for_time_step(states))
        )

    def of_states(self, states: HistoryT, /) -> NumPyObstaclePositions:
        return NumPyAdaptedObstaclePositions(np.asarray(self.inner.of_states(states)))


@dataclass(kw_only=True)
class OrientationExtractorAdapter[StatesT, HistoryT](
    NumPyObstacleOrientationExtractor[
        StatesT,
        HistoryT,
        NumPyObstacleOrientationsForTimeStep,
        NumPyObstacleOrientations,
    ]
):
    """Adapts a JAX orientation extractor to the NumPy interface for obstacle assignment."""

    inner: JaxObstacleOrientationExtractor[
        StatesT,
        HistoryT,
        JaxObstacleOrientationsForTimeStep,
        JaxObstacleOrientations,
    ]

    @staticmethod
    def adapt(
        extractor: JaxOrientationsExtractor[StatesT, HistoryT],
    ) -> "OrientationExtractorAdapter[StatesT, HistoryT]":
        return OrientationExtractorAdapter(inner=extractor)

    def of_states_for_time_step(
        self, states: StatesT, /
    ) -> NumPyObstacleOrientationsForTimeStep:
        return NumPyAdaptedObstacleOrientationsForTimeStep(
            np.asarray(self.inner.of_states_for_time_step(states))
        )

    def of_states(self, states: HistoryT, /) -> NumPyObstacleOrientations:
        return NumPyAdaptedObstacleOrientations(
            np.asarray(self.inner.of_states(states))
        )


@dataclass(frozen=True)
class JaxHungarianObstacleIdAssignment[
    StatesT: ObstacleStatesForTimeStep,
    HistoryT: ObstacleStatesHistory,
](ObstacleIdAssignment[StatesT, JaxObstacleIds, HistoryT]):
    """Matches detected obstacles to tracked IDs using the Hungarian algorithm
    on pose differences."""

    # NOTE: Internally the Hungarian assignment is still done using SciPy and NumPy.
    inner: NumPyHungarianObstacleIdAssignment[StatesT, HistoryT]

    @staticmethod
    def create[S: ObstacleStatesForTimeStep, H: ObstacleStatesHistory](
        *,
        position_extractor: JaxPositionsExtractor[S, H],
        orientation_extractor: JaxOrientationsExtractor[S, H] | None = None,
        cutoff: float,
        orientation_cutoff: float | None = None,
        start_id: int = 0,
    ) -> "JaxHungarianObstacleIdAssignment":
        return JaxHungarianObstacleIdAssignment(
            inner=NumPyHungarianObstacleIdAssignment.create(
                position_extractor=PositionExtractorAdapter.adapt(position_extractor),
                orientation_extractor=(
                    OrientationExtractorAdapter.adapt(orientation_extractor)
                    if orientation_extractor is not None
                    else None
                ),
                cutoff=cutoff,
                orientation_cutoff=orientation_cutoff,
                start_id=start_id,
            )
        )

    def __call__(
        self, states: StatesT, /, *, history: HistoryT, ids: JaxObstacleIds
    ) -> JaxObstacleIds:
        return JaxObstacleIds.create(
            ids=jnp.asarray(
                self.inner(
                    states, history=history, ids=NumPyObstacleIds(np.asarray(ids))
                )
            )
        )
