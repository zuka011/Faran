from typing import Final, Sequence

from faran import types

from jaxtyping import Array as JaxArray, Float, Num
from numtypes import Array

import numpy as np
import jax.numpy as jnp

D_O: Final = types.obstacle.POSE_D_O

type D_o = types.obstacle.PoseD_o
type NumPyState = types.numpy.simple.State
type NumPyStateBatch = types.numpy.simple.StateBatch
type NumPyControlInputSequence = types.numpy.simple.ControlInputSequence
type NumPyControlInputBatch = types.numpy.simple.ControlInputBatch
type NumPySampledObstacle2dPoses = types.numpy.SampledObstacle2dPoses
type NumPyObstacleIds = types.numpy.ObstacleIds
type NumPyObstacle2dPosesForTimeStep = types.numpy.Obstacle2dPosesForTimeStep
type NumPyObstacle2dPoses = types.numpy.Obstacle2dPoses
type NumPySimpleObstacleStates = types.numpy.simple.ObstacleStates
type NumPyDistance = types.numpy.Distance
type NumPyBoundaryDistance = types.numpy.BoundaryDistance

type JaxState = types.jax.simple.State
type JaxStateBatch = types.jax.simple.StateBatch
type JaxControlInputSequence = types.jax.simple.ControlInputSequence
type JaxControlInputBatch = types.jax.simple.ControlInputBatch
type JaxSampledObstacle2dPoses = types.jax.SampledObstacle2dPoses
type JaxObstacleIds = types.jax.ObstacleIds
type JaxObstacle2dPosesForTimeStep = types.jax.Obstacle2dPosesForTimeStep
type JaxObstacle2dPoses = types.jax.Obstacle2dPoses
type JaxSimpleObstacleStates = types.jax.simple.ObstacleStates
type JaxDistance = types.jax.Distance
type JaxBoundaryDistance = types.jax.BoundaryDistance


class numpy:
    @staticmethod
    def state(array: Float[Array, " D_x"]) -> NumPyState:
        return types.numpy.simple.state(array)

    @staticmethod
    def state_batch(
        array: Float[Array, "T D_x M"],
    ) -> NumPyStateBatch:
        return types.numpy.simple.state_batch(array)

    @staticmethod
    def control_input_sequence(
        array: Float[Array, "T D_u"],
    ) -> NumPyControlInputSequence:
        return types.numpy.simple.control_input_sequence(array)

    @staticmethod
    def control_input_batch(
        array: Float[Array, "T D_u M"],
    ) -> NumPyControlInputBatch:
        return types.numpy.simple.control_input_batch(array)

    @staticmethod
    def obstacle_ids(
        array: Num[Array, " K"] | Sequence[int],
    ) -> NumPyObstacleIds:
        return types.numpy.obstacle_ids.create(ids=np.asarray(array))

    @staticmethod
    def simple_obstacle_states(
        *,
        states: Float[Array, "T D_o K"],
        covariance: Float[Array, "T D_o D_o K"] | None = None,
    ) -> NumPySimpleObstacleStates:
        return types.numpy.simple.obstacle_states.create(
            states=states, covariance=covariance
        )

    @staticmethod
    def simple_obstacle_states_for_time_step(
        *, array: Float[Array, " D_o K"]
    ) -> NumPySimpleObstacleStates:
        return types.numpy.simple.obstacle_states_for_time_step.create(array=array)

    @staticmethod
    def obstacle_2d_poses(
        *,
        x: Float[Array, "T K"],
        y: Float[Array, "T K"],
        heading: Float[Array, "T K"] | None = None,
        covariance: Float[Array, "T D_o D_o K"] | None = None,
    ) -> NumPyObstacle2dPoses:
        return types.numpy.obstacle_2d_poses.create(
            x=x,
            y=y,
            heading=heading if heading is not None else np.zeros_like(x),
            covariance=covariance,
        )

    @staticmethod
    def obstacle_2d_poses_for_time_step(
        *,
        x: Float[Array, " K"],
        y: Float[Array, " K"],
        heading: Float[Array, " K"] | None = None,
    ) -> NumPyObstacle2dPosesForTimeStep:
        return types.numpy.obstacle_2d_poses_for_time_step.create(
            x=x,
            y=y,
            heading=heading if heading is not None else np.zeros_like(x),
        )

    @staticmethod
    def obstacle_2d_pose_samples(
        *,
        x: Float[Array, "T K N"],
        y: Float[Array, "T K N"],
        heading: Float[Array, "T K N"] | None = None,
    ) -> NumPySampledObstacle2dPoses:
        return types.numpy.obstacle_2d_poses.sampled(
            x=x,
            y=y,
            heading=heading if heading is not None else np.zeros_like(x),
        )

    @staticmethod
    def distance(
        array: Float[Array, "T V M N"],
    ) -> NumPyDistance:
        return types.numpy.distance(array)

    @staticmethod
    def boundary_distance(
        array: Float[Array, "T M"],
    ) -> NumPyBoundaryDistance:
        return types.numpy.boundary_distance(array)


class jax:
    @staticmethod
    def state(array: Float[Array, " D_x"]) -> JaxState:
        return types.jax.simple.state.create(array=array)

    @staticmethod
    def obstacle_ids(
        array: Num[Array, " K"] | Sequence[int],
    ) -> JaxObstacleIds:
        return types.jax.obstacle_ids.create(ids=np.asarray(array))

    @staticmethod
    def state_batch(
        array: Float[Array, "T D_x M"],
    ) -> JaxStateBatch:
        return types.jax.simple.state_batch.wrap(array=array)

    @staticmethod
    def control_input_sequence(
        array: Float[Array, "T D_u"],
    ) -> JaxControlInputSequence:
        return types.jax.simple.control_input_sequence.create(array=array)

    @staticmethod
    def control_input_batch(
        array: Float[Array, "T D_u M"] | Float[JaxArray, "T D_u M"],
    ) -> JaxControlInputBatch:
        return types.jax.simple.control_input_batch.create(array=array)

    @staticmethod
    def simple_obstacle_states(
        *,
        states: Float[Array, "T D_o K"] | Float[JaxArray, "T D_o K"],
        covariance: Float[Array, "T D_o D_o K"]
        | Float[JaxArray, "T D_o D_o K"]
        | None = None,
    ) -> JaxSimpleObstacleStates:
        return types.jax.simple.obstacle_states.create(
            states=states, covariance=covariance
        )

    @staticmethod
    def simple_obstacle_states_for_time_step(
        *, array: Float[Array, " D_o K"] | Float[JaxArray, " D_o K"]
    ) -> JaxSimpleObstacleStates:
        return types.jax.simple.obstacle_states_for_time_step.create(array=array)

    @staticmethod
    def obstacle_2d_poses(
        *,
        x: Float[Array, "T K"] | Float[JaxArray, "T K"],
        y: Float[Array, "T K"] | Float[JaxArray, "T K"],
        heading: Float[Array, "T K"] | Float[JaxArray, "T K"] | None = None,
        covariance: Float[Array, "T D_o D_o K"]
        | Float[JaxArray, f"T {D_O} {D_O} K"]
        | None = None,
    ) -> JaxObstacle2dPoses:
        return types.jax.obstacle_2d_poses.create(
            x=x,
            y=y,
            heading=heading if heading is not None else jnp.zeros_like(x),
            covariance=covariance,
        )

    @staticmethod
    def obstacle_2d_poses_for_time_step(
        *,
        x: Float[Array, " K"] | Float[JaxArray, " K"],
        y: Float[Array, " K"] | Float[JaxArray, " K"],
        heading: Float[Array, " K"] | Float[JaxArray, " K"] | None = None,
    ) -> JaxObstacle2dPosesForTimeStep:
        return types.jax.obstacle_2d_poses_for_time_step.create(
            x=x,
            y=y,
            heading=heading if heading is not None else jnp.zeros_like(x),
        )

    @staticmethod
    def obstacle_2d_pose_samples(
        *,
        x: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
        y: Float[Array, "T K N"] | Float[JaxArray, "T K N"],
        heading: Float[Array, "T K N"] | Float[JaxArray, "T K N"] | None = None,
    ) -> JaxSampledObstacle2dPoses:
        return types.jax.obstacle_2d_poses.sampled(
            x=x,
            y=y,
            heading=heading if heading is not None else jnp.zeros_like(x),
        )

    @staticmethod
    def distance(
        array: Float[Array, "T V M N"] | Float[JaxArray, "T V M N"],
    ) -> JaxDistance:
        return types.jax.distance.create(array=array)

    @staticmethod
    def boundary_distance(
        array: Float[Array, "T M"] | Float[JaxArray, "T M"],
    ) -> JaxBoundaryDistance:
        return types.jax.boundary_distance.create(array=array)
