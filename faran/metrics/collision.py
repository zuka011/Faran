from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
    Distance,
    DistanceExtractor,
    StateSequence,
    ObstacleStates,
    SimulationData,
    Metric,
)
from faran.collectors import access

from jaxtyping import Float, Bool

import numpy as np


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class CollisionMetricResult:
    """Results of the collision metric, including distances and collision flags."""

    distances: Float[Array, "T V"]
    distance_threshold: float

    @cached_property
    def min_distances(self) -> Float[Array, " V"]:
        return self.distances.min(axis=0)

    @cached_property
    def collisions(self) -> Bool[Array, "T V"]:
        return self.distances <= self.distance_threshold

    @cached_property
    def collision_detected(self) -> bool:
        return bool(self.collisions.any())


@dataclass(kw_only=True, frozen=True)
class CollisionMetric[StateBatchT, SampledObstacleStatesT](
    Metric[CollisionMetricResult]
):
    """Metric evaluating minimum distances between the ego vehicle and obstacles."""

    distance: DistanceExtractor[StateBatchT, SampledObstacleStatesT, Distance]
    distance_threshold: float

    @staticmethod
    def create[S, SOS](
        *,
        distance_threshold: float,
        distance: DistanceExtractor[S, SOS, Distance],
    ) -> "CollisionMetric":
        return CollisionMetric(distance=distance, distance_threshold=distance_threshold)

    def compute(self, data: SimulationData) -> CollisionMetricResult:
        states = data(access.states.assume(StateSequence[StateBatchT]).require())
        obstacle_states = data(
            access.obstacle_states.assume(
                ObstacleStates[SampledObstacleStatesT]
            ).require()
        )

        measured_distances = self.distance(
            states=states.batched(), obstacle_states=obstacle_states.single()
        )

        distances = np.asarray(measured_distances).reshape(states.horizon, -1)

        return CollisionMetricResult(
            distances=distances, distance_threshold=self.distance_threshold
        )

    @property
    def name(self) -> str:
        return "collision"
