from typing import overload
from dataclasses import dataclass

from faran.types import (
    DistanceExtractor,
    Array,
    NumPyHeadings,
    NumPyPositions,
    NumPyPositionExtractor,
    NumPyHeadingExtractor,
    NumPySampledObstaclePositions,
    NumPySampledObstacleHeadings,
    NumPySampledObstaclePositionExtractor,
    NumPySampledObstacleHeadingExtractor,
)
from faran.costs.collision import NumPyDistance
from faran.costs.distance.basic import replace_missing
from faran.costs.distance.circles.common import Circles

from jaxtyping import Float

import numpy as np


type OriginsArray = Float[Array, "N 2"]
type RadiiArray = Float[Array, " N"]


@dataclass(frozen=True)
class NumPyCircleDistanceExtractor[StateT, SampledObstacleStatesT](
    DistanceExtractor[StateT, SampledObstacleStatesT, NumPyDistance]
):
    """
    Computes the distances between parts of the ego robot and obstacles. Both the ego
    and the obstacles are represented as collections of circles.
    """

    ego: Circles
    obstacle: Circles
    positions_from: NumPyPositionExtractor[StateT]
    headings_from: NumPyHeadingExtractor[StateT]
    obstacle_positions_from: NumPySampledObstaclePositionExtractor[
        SampledObstacleStatesT
    ]
    obstacle_headings_from: NumPySampledObstacleHeadingExtractor[SampledObstacleStatesT]

    @staticmethod
    def create[S, SOS](
        *,
        ego: Circles,
        obstacle: Circles,
        position_extractor: NumPyPositionExtractor[S],
        heading_extractor: NumPyHeadingExtractor[S],
        obstacle_position_extractor: NumPySampledObstaclePositionExtractor[SOS],
        obstacle_heading_extractor: NumPySampledObstacleHeadingExtractor[SOS],
    ) -> "NumPyCircleDistanceExtractor[S, SOS]":
        return NumPyCircleDistanceExtractor(
            ego=ego,
            obstacle=obstacle,
            positions_from=position_extractor,
            headings_from=heading_extractor,
            obstacle_positions_from=obstacle_position_extractor,
            obstacle_headings_from=obstacle_heading_extractor,
        )

    def __call__(
        self, *, states: StateT, obstacle_states: SampledObstacleStatesT
    ) -> NumPyDistance:
        obstacle_positions, obstacle_headings = replace_missing(
            positions=self.obstacle_positions_from(obstacle_states),
            headings=self.obstacle_headings_from(obstacle_states),
        )

        return NumPyDistance(
            compute_circle_distances(
                ego_positions=self.positions_from(states),
                ego_headings=self.headings_from(states),
                ego=self.ego,
                obstacle_positions=obstacle_positions,
                obstacle_headings=obstacle_headings,
                obstacle=self.obstacle,
            )
        )


def compute_circle_distances(
    *,
    ego_positions: NumPyPositions,
    ego_headings: NumPyHeadings,
    ego: Circles,
    obstacle_positions: NumPySampledObstaclePositions,
    obstacle_headings: NumPySampledObstacleHeadings,
    obstacle: Circles,
) -> Float[Array, "T V M N"]:
    ego_global_x, ego_global_y = to_global_positions(
        x=ego_positions.x(),
        y=ego_positions.y(),
        heading=ego_headings.heading(),
        local_origins=ego.origins,
    )

    obstacle_global_x, obstacle_global_y = to_global_positions(
        x=obstacle_positions.x(),
        y=obstacle_positions.y(),
        heading=obstacle_headings.heading(),
        local_origins=obstacle.origins,
    )

    pairwise_distances = pairwise_min_distances(
        ego_x=ego_global_x,
        ego_y=ego_global_y,
        ego_radii=ego.radii,
        obstacle_x=obstacle_global_x,
        obstacle_y=obstacle_global_y,
        obstacle_radii=obstacle.radii,
    )

    return min_distance_per_ego_part(pairwise_distances)


@overload
def to_global_positions(
    *,
    x: Float[Array, "T M"],
    y: Float[Array, "T M"],
    heading: Float[Array, "T M"],
    local_origins: OriginsArray,
) -> tuple[Float[Array, "V T M"], Float[Array, "V T M"]]: ...


@overload
def to_global_positions(  # pyright: ignore[reportOverlappingOverload]
    *,
    x: Float[Array, "T K N"],
    y: Float[Array, "T K N"],
    heading: Float[Array, "T K N"],
    local_origins: OriginsArray,
) -> tuple[Float[Array, "C T K N"], Float[Array, "C T K N"]]: ...


def to_global_positions(
    *,
    x: Float[Array, "T M"] | Float[Array, "T K N"],
    y: Float[Array, "T M"] | Float[Array, "T K N"],
    heading: Float[Array, "T M"] | Float[Array, "T K N"],
    local_origins: OriginsArray | OriginsArray,
) -> (
    tuple[Float[Array, "V T M"], Float[Array, "V T M"]]
    | tuple[Float[Array, "C T K N"], Float[Array, "C T K N"]]
):
    local_xy = local_origins.reshape((-1, 2) + (1,) * x.ndim)
    local_x, local_y = local_xy[:, 0], local_xy[:, 1]

    cos_h, sin_h = np.cos(heading), np.sin(heading)

    return (
        x + local_x * cos_h - local_y * sin_h,
        y + local_x * sin_h + local_y * cos_h,
    )


def pairwise_min_distances(
    ego_x: Float[Array, "V T M"],
    ego_y: Float[Array, "V T M"],
    ego_radii: RadiiArray,
    obstacle_x: Float[Array, "C T K N"],
    obstacle_y: Float[Array, "C T K N"],
    obstacle_radii: RadiiArray,
) -> Float[Array, "V C T M K N"]:
    dx = (
        ego_x[:, np.newaxis, :, :, np.newaxis, np.newaxis]
        - obstacle_x[np.newaxis, :, :, np.newaxis, :, :]
    )
    dy = (
        ego_y[:, np.newaxis, :, :, np.newaxis, np.newaxis]
        - obstacle_y[np.newaxis, :, :, np.newaxis, :, :]
    )

    center_distances = np.sqrt(dx**2 + dy**2)
    radii_sum = (
        ego_radii[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        + obstacle_radii[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    )

    return center_distances - radii_sum


def min_distance_per_ego_part(
    pairwise_distances: Float[Array, "V C T M K N"],
) -> Float[Array, "T V M N"]:
    V, C, T, M, K, N = pairwise_distances.shape

    if C == 0 or K == 0:
        return np.full((T, V, M, N), np.inf)

    min_over_obstacles = np.min(pairwise_distances, axis=(1, 4))
    return np.transpose(min_over_obstacles, (1, 0, 2, 3))
