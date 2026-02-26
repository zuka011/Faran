from typing import cast
from dataclasses import dataclass
from functools import cached_property

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
    JaxCosts,
    JaxObstacleStateProvider,
    JaxObstacleStateSampler,
    JaxDistanceExtractor,
    JaxRisk,
    JaxRiskMetric,
)
from faran.states import JaxSimpleCosts

from jaxtyping import Array as JaxArray, Float, Scalar

import numpy as np
import jax
import jax.numpy as jnp


@jaxtyped
@dataclass(frozen=True)
class JaxDistance(Distance):
    """Pairwise distances between V vehicle parts and N obstacle samples over T time steps and M rollouts."""

    _array: Float[JaxArray, "T V M N"]

    @staticmethod
    def create(
        *, array: Float[Array, "T V M N"] | Float[JaxArray, "T V M N"]
    ) -> "JaxDistance":
        """Creates a JAX distance from the given array."""
        return JaxDistance(jnp.asarray(array))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T V M N"]:
        return self._numpy_array

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
    def array(self) -> Float[JaxArray, "T V M N"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T V M N"]:
        return np.asarray(self._array)


class JaxNoMetric:
    """Bypasses risk computation, evaluating collision cost with a single deterministic sample."""

    @staticmethod
    def create() -> "JaxNoMetric":
        return JaxNoMetric()

    def compute[StateT, ObstacleStateT, SampledObstacleStateT](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> JaxRisk:
        samples = sampler(obstacle_states, count=1)
        return JaxRisk(cost_function(states=states, samples=samples).squeeze(axis=-1))

    @property
    def name(self) -> str:
        return "No Metric"


@dataclass(kw_only=True, frozen=True)
class JaxCollisionCost[
    StateT,
    ObstacleStatesT: ObstacleStates,
    SampledObstacleStatesT,
    DistanceT: JaxDistance,
](CostFunction[ControlInputBatch, StateT, JaxCosts]):
    """Collision avoidance cost based on distance thresholds to sampled obstacle positions."""

    obstacle_states: JaxObstacleStateProvider[ObstacleStatesT]
    sampler: JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT]
    distance: JaxDistanceExtractor[StateT, SampledObstacleStatesT, DistanceT]
    distance_threshold: Float[JaxArray, " V"]
    weight: float
    metric: JaxRiskMetric[StateT, ObstacleStatesT, SampledObstacleStatesT]

    @staticmethod
    def create[S, OS: ObstacleStates, SOS, D: JaxDistance](
        *,
        obstacle_states: JaxObstacleStateProvider[OS],
        sampler: JaxObstacleStateSampler[OS, SOS],
        distance: JaxDistanceExtractor[S, SOS, D],
        distance_threshold: Float[Array, " V"],
        weight: float,
        metric: JaxRiskMetric[S, OS, SOS] | None = None,
    ) -> "JaxCollisionCost[S, OS, SOS, D]":
        return JaxCollisionCost(
            obstacle_states=obstacle_states,
            sampler=sampler,
            distance=distance,
            distance_threshold=jnp.asarray(distance_threshold),
            weight=weight,
            metric=metric
            if metric is not None
            else cast(JaxRiskMetric[S, OS, SOS], JaxNoMetric()),
        )

    def __call__(self, *, inputs: ControlInputBatch, states: StateT) -> JaxCosts:
        def cost(
            *, states: StateT, samples: SampledObstacleStatesT
        ) -> Float[JaxArray, "T M N"]:
            return collision_cost(
                distance=self.distance(states=states, obstacle_states=samples).array,
                distance_threshold=self.distance_threshold,
                weight=self.weight,
            )

        return JaxSimpleCosts(
            jnp.zeros((inputs.horizon, inputs.rollout_count))
            if (obstacle_states := self.obstacle_states()).count == 0
            else self.metric.compute(
                cost,
                states=states,
                obstacle_states=obstacle_states,
                sampler=self.sampler,
            ).array
        )


@jax.jit
@jaxtyped
def collision_cost(
    *,
    distance: Float[JaxArray, "T V M N"],
    distance_threshold: Float[JaxArray, " V"],
    weight: Scalar,
) -> Float[JaxArray, "T M N"]:
    cost = distance_threshold[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] - distance
    return weight * jnp.clip(cost, 0, None).sum(axis=1)
