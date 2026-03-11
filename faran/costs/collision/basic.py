from typing import cast
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    Array,
    DataType,
    Distance,
    ControlInputBatch,
    CostFunction,
    ObstacleStates,
    ObstacleStateSampler,
    SampleCostFunction,
    NumPyCosts,
    NumPyObstacleStateProvider,
    NumPyObstacleStateSampler,
    NumPyDistanceExtractor,
    NumPyRisk,
    NumPyRiskMetric,
)
from faran.states import NumPySimpleCosts

from numtypes import D
from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(frozen=True)
class NumPyDistance(Distance):
    """Pairwise distances between V vehicle parts and N obstacle samples over T time steps and M rollouts."""

    _array: Float[Array, "T V M N"]

    @staticmethod
    def create(array: Float[Array, "T V M N"]) -> "NumPyDistance":
        """Creates a NumPy distance from the given array."""
        return NumPyDistance(array)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T V M N"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self._array.shape[0]

    @property
    def vehicle_parts(self) -> int:
        return self._array.shape[1]

    @property
    def rollout_count(self) -> int:
        return self._array.shape[2]

    @property
    def sample_count(self) -> int:
        return self._array.shape[3]

    @property
    def array(self) -> Float[Array, "T V M N"]:
        return self._array


class NumPyNoMetric:
    """Bypasses risk computation, evaluating collision cost with a single deterministic sample."""

    @staticmethod
    def create() -> "NumPyNoMetric":
        return NumPyNoMetric()

    def compute[StateT, ObstacleStateT, SampledObstacleStateT](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Float[Array, "T M _"]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> NumPyRisk:
        samples = sampler(obstacle_states, count=1)
        return NumPyRisk(cost_function(states=states, samples=samples).squeeze(axis=-1))

    @property
    def name(self) -> str:
        return "No Metric"


@dataclass(kw_only=True, frozen=True)
class NumPyCollisionCost[
    StateT,
    ObstacleStatesT: ObstacleStates,
    SampledObstacleStatesT,
    DistanceT: NumPyDistance,
](CostFunction[ControlInputBatch, StateT, NumPyCosts]):
    """Collision avoidance cost based on distance thresholds to sampled obstacle positions."""

    obstacle_states: NumPyObstacleStateProvider[ObstacleStatesT]
    sampler: NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT]
    distance: NumPyDistanceExtractor[StateT, SampledObstacleStatesT, DistanceT]
    distance_threshold: Float[Array, " V"]
    weight: float
    metric: NumPyRiskMetric[StateT, ObstacleStatesT, SampledObstacleStatesT]

    @staticmethod
    def create[S, OS: ObstacleStates, SOS, D: NumPyDistance](
        *,
        obstacle_states: NumPyObstacleStateProvider[OS],
        sampler: NumPyObstacleStateSampler[OS, SOS],
        distance: NumPyDistanceExtractor[S, SOS, D],
        distance_threshold: Float[Array, " V"],
        weight: float,
        metric: NumPyRiskMetric[S, OS, SOS] | None = None,
    ) -> "NumPyCollisionCost[S, OS, SOS, D]":
        return NumPyCollisionCost(
            obstacle_states=obstacle_states,
            sampler=sampler,
            distance=distance,
            distance_threshold=distance_threshold,
            weight=weight,
            metric=metric
            if metric is not None
            else cast(NumPyRiskMetric[S, OS, SOS], NumPyNoMetric()),
        )

    def __call__(self, *, inputs: ControlInputBatch, states: StateT) -> NumPyCosts:
        def cost(
            *, states: StateT, samples: SampledObstacleStatesT
        ) -> Float[Array, "T M _"]:
            cost = (
                self.distance_threshold[np.newaxis, :, np.newaxis, np.newaxis]
                - self.distance(states=states, obstacle_states=samples).array
            )

            return np.clip(cost, 0, None).sum(axis=1)

        horizon, rollouts = inputs.horizon, inputs.rollout_count

        return NumPySimpleCosts(
            (
                np.zeros((horizon, rollouts))
                if (obstacle_states := self.obstacle_states()).count == 0
                else self.weight
                * self.metric.compute(
                    cost,
                    states=states,
                    obstacle_states=obstacle_states,
                    sampler=self.sampler,
                ).array
            )
        )
