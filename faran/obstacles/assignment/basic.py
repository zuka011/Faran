from typing import Final, NamedTuple, cast
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    Array,
    DataType,
    ObstacleIdAssignment,
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    NumPyObstaclePositionsForTimeStep,
    NumPyObstaclePositions,
    NumPyObstaclePositionExtractor,
    NumPyObstacleOrientationsForTimeStep,
    NumPyObstacleOrientations,
    NumPyObstacleOrientationExtractor,
)
from faran.obstacles.history import NumPyObstacleIds

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numtypes import D
from jaxtyping import Float, Int, Bool

import numpy as np


type PositionsExtractor[StatesT, HistoryT] = NumPyObstaclePositionExtractor[
    StatesT, HistoryT, NumPyObstaclePositionsForTimeStep, NumPyObstaclePositions
]

type OrientationsExtractor[StatesT, HistoryT] = NumPyObstacleOrientationExtractor[
    StatesT,
    HistoryT,
    NumPyObstacleOrientationsForTimeStep,
    NumPyObstacleOrientations,
]


class WithCutoff[T](NamedTuple):
    extractor: T
    cutoff: float


@jaxtyped
@dataclass(frozen=True)
class NoOrientationsForTimeStep(NumPyObstacleOrientationsForTimeStep):
    _array: Float[Array, "0 K"]

    @staticmethod
    def create(*, obstacle_count: int) -> "NoOrientationsForTimeStep":
        return NoOrientationsForTimeStep(np.zeros((0, obstacle_count)))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "0 K"]:
        return self.array

    @property
    def dimension(self) -> D[0]:
        return 0

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "0 K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NoOrientations(NumPyObstacleOrientations):
    _array: Float[Array, "T 0 K"]

    @staticmethod
    def create(*, horizon: int, obstacle_count: int) -> "NoOrientations":
        return NoOrientations(np.zeros((horizon, 0, obstacle_count)))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 0 K"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> D[0]:
        return 0

    @property
    def count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T 0 K"]:
        return self._array


class NoOrientationsExtractor:
    def of_states_for_time_step(
        self, states: ObstacleStatesForTimeStep, /
    ) -> NumPyObstacleOrientationsForTimeStep:
        return NoOrientationsForTimeStep.create(obstacle_count=states.count)

    def of_states(self, states: ObstacleStatesHistory, /) -> NumPyObstacleOrientations:
        return NoOrientations.create(
            horizon=states.horizon, obstacle_count=states.count
        )


@dataclass(kw_only=True)
class NumPyHungarianObstacleIdAssignment[
    StatesT: ObstacleStatesForTimeStep,
    HistoryT: ObstacleStatesHistory,
](ObstacleIdAssignment[StatesT, NumPyObstacleIds, HistoryT]):
    """Matches detected obstacles to tracked IDs using the Hungarian algorithm on pose differences."""

    positions: Final[WithCutoff[PositionsExtractor[StatesT, HistoryT]]]
    orientations: Final[WithCutoff[OrientationsExtractor[StatesT, HistoryT]]]
    next_id: int

    @staticmethod
    def create[S: ObstacleStatesForTimeStep, H: ObstacleStatesHistory](
        *,
        position_extractor: PositionsExtractor[S, H],
        orientation_extractor: OrientationsExtractor[S, H] | None = None,
        cutoff: float,
        orientation_cutoff: float | None = None,
        start_id: int = 0,
    ) -> "NumPyHungarianObstacleIdAssignment":
        assert (orientation_extractor is None) == (orientation_cutoff is None), (
            f"The orientation cutoff must be specified only if an orientation extractor is provided. "
            f"Got: {orientation_extractor} (extractor) and {orientation_cutoff} (cutoff)."
        )

        return NumPyHungarianObstacleIdAssignment(
            positions=WithCutoff(position_extractor, cutoff),
            orientations=WithCutoff(NoOrientationsExtractor(), 1.0)
            if orientation_extractor is None or orientation_cutoff is None
            else WithCutoff(orientation_extractor, orientation_cutoff),
            next_id=start_id,
        )

    def __call__(
        self, states: StatesT, /, *, history: HistoryT, ids: NumPyObstacleIds
    ) -> NumPyObstacleIds:
        if states.count == 0:
            return NumPyObstacleIds.empty()

        if history.horizon == 0:
            return NumPyObstacleIds.create(ids=self._allocate_ids(states.count))

        last_positions, last_orientations, valid_ids = self._valid_history_from(
            history, ids
        )

        if len(valid_ids) == 0:
            return NumPyObstacleIds.create(ids=self._allocate_ids(states.count))

        current_indices, history_indices, matched = self._matching_for(
            current_positions=(
                self.positions.extractor.of_states_for_time_step(states).array
            ),
            last_positions=last_positions,
            current_orientations=(
                self.orientations.extractor.of_states_for_time_step(states).array
            ),
            last_orientations=last_orientations,
        )

        return NumPyObstacleIds.create(
            ids=self._assign_ids(
                current_obstacle_count=states.count,
                current_indices=current_indices,
                history_indices=history_indices,
                matched=matched,
                valid_ids=valid_ids,
            )
        )

    def _valid_history_from(
        self, history: HistoryT, ids: NumPyObstacleIds
    ) -> tuple[Float[Array, "D_p K"], Float[Array, "D_o K"], Int[Array, " K"]]:
        id_count = ids.count
        positions = self.positions.extractor.of_states(history).array
        orientations = self.orientations.extractor.of_states(history).array

        # NOTE: The history may be padded with states for more obstacles
        # than there are IDs.
        last_positions = positions[-1, :, :id_count]
        last_orientations = orientations[-1, :, :id_count]

        # NOTE: Checking just the first dimension for nan is sufficient
        valid = ~np.isnan(last_positions[0])

        return last_positions[:, valid], last_orientations[:, valid], ids.array[valid]

    def _matching_for(
        self,
        *,
        current_positions: Float[Array, "D_p K_c"],
        last_positions: Float[Array, "D_p K_h"],
        current_orientations: Float[Array, "D_o K_c"],
        last_orientations: Float[Array, "D_o K_h"],
    ) -> tuple[Int[Array, " M"], Int[Array, " M"], Bool[Array, " M"]]:
        large_value = max(self.positions.cutoff, self.orientations.cutoff) + 1e9
        position_distances = cdist(current_positions.T, last_positions.T)
        orientation_distances = angular_distance(
            current_orientations, last_orientations
        )

        # NOTE: We set a large value for distances beyond the cutoff
        # to prevent them from being matched.
        distances = position_distances
        distances[position_distances > self.positions.cutoff] = large_value
        distances[orientation_distances > self.orientations.cutoff] = large_value

        current_indices, history_indices = linear_sum_assignment(distances)
        matched = distances[current_indices, history_indices] < large_value

        return current_indices, history_indices, matched

    def _assign_ids(
        self,
        *,
        current_obstacle_count: int,
        current_indices: Int[Array, " M"],
        history_indices: Int[Array, " M"],
        matched: Bool[Array, " M"],
        valid_ids: Int[Array, " N_valid"],
    ) -> Int[Array, " K_c"]:
        result = np.full(current_obstacle_count, -1, dtype=np.int64)
        result[current_indices[matched]] = valid_ids[history_indices[matched]]

        unmatched = result == -1
        result[unmatched] = self._allocate_ids(np.sum(unmatched))

        return result

    def _allocate_ids(self, count: int) -> Int[Array, " K"]:
        new_ids = np.arange(self.next_id, self.next_id + count)
        self.next_id += count

        return cast(Int[Array, " K"], new_ids)


def angular_distance(
    a: Float[Array, "D_o K_c"], b: Float[Array, "D_o K_h"]
) -> Float[Array, "K_c K_h"]:
    assert (D_o := a.shape[0]) <= 1, (
        f"Multi-dimensional orientations (D_o={D_o}) are not yet supported."
    )

    return np.linalg.norm(
        np.minimum(
            delta := np.abs(a[:, :, np.newaxis] - b[:, np.newaxis, :]),
            2 * np.pi - delta,
        ),
        axis=0,
    )
