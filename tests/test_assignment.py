from faran import ObstacleIdAssignment, ObstacleIds, types, obstacles

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


type NumPyObstacleStatesForTimeStep = types.numpy.Obstacle2dPosesForTimeStep
type NumPyObstacleStates = types.numpy.Obstacle2dPoses
type NumPyObstacle2dPositionsForTimeStep = types.numpy.Obstacle2dPositionsForTimeStep
type NumPyObstacle2dPositions = types.numpy.Obstacle2dPositions
type NumPyObstacleHeadingsForTimeStep = types.numpy.ObstacleHeadingsForTimeStep
type NumPyObstacleHeadings = types.numpy.ObstacleHeadings
type JaxObstacleStatesForTimeStep = types.jax.Obstacle2dPosesForTimeStep
type JaxObstacleStates = types.jax.Obstacle2dPoses
type JaxObstacle2dPositionsForTimeStep = types.jax.Obstacle2dPositionsForTimeStep
type JaxObstacle2dPositions = types.jax.Obstacle2dPositions
type JaxObstacleHeadingsForTimeStep = types.jax.ObstacleHeadingsForTimeStep
type JaxObstacleHeadings = types.jax.ObstacleHeadings


class NumPyObstaclePositionExtractor:
    def of_states_for_time_step(
        self, states: NumPyObstacleStatesForTimeStep, /
    ) -> NumPyObstacle2dPositionsForTimeStep:
        return states.positions()

    def of_states(self, states: NumPyObstacleStates, /) -> NumPyObstacle2dPositions:
        return states.positions()


class JaxObstaclePositionExtractor:
    def of_states_for_time_step(
        self, states: JaxObstacleStatesForTimeStep, /
    ) -> JaxObstacle2dPositionsForTimeStep:
        return states.positions()

    def of_states(self, states: JaxObstacleStates, /) -> JaxObstacle2dPositions:
        return states.positions()


class NumPyObstacleHeadingExtractor:
    def of_states_for_time_step(
        self, states: NumPyObstacleStatesForTimeStep, /
    ) -> NumPyObstacleHeadingsForTimeStep:
        return states.headings()

    def of_states(self, states: NumPyObstacleStates, /) -> NumPyObstacleHeadings:
        return states.headings()


class JaxObstacleHeadingExtractor:
    def of_states_for_time_step(
        self, states: JaxObstacleStatesForTimeStep, /
    ) -> JaxObstacleHeadingsForTimeStep:
        return states.headings()

    def of_states(self, states: JaxObstacleStates, /) -> JaxObstacleHeadings:
        return states.headings()


class test_that_ids_are_assigned_to_obstacles:
    def cases(id_assignment, position_extractor, heading_extractor, data) -> None:
        return [
            (  # All new obstacles (no history)
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.2, 1.5, 3.0], shape=(K := 3,)),
                    y=array([1.2, 2.5, 4.0], shape=(K,)),
                    heading=array([1.0, 2.0, 1.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=np.empty((0, 0)),
                    y=np.empty((0, 0)),
                    heading=np.empty((0, 0)),
                ),
                ids := data.obstacle_ids([]),
                expected := data.obstacle_ids([1, 2, 3]),
            ),
            (  # Single obstacle is tracked
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.3], shape=(K := 1,)),
                    y=array([1.3], shape=(K,)),
                    heading=array([1.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2]], shape=(1, 1)),
                    y=array([[1.2]], shape=(1, 1)),
                    heading=array([[1.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([5]),
            ),
            (  # Single obstacle outside cutoff should get a new ID
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([5.0], shape=(K := 1,)),
                    y=array([5.0], shape=(K,)),
                    heading=array([1.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2]], shape=(1, 1)),
                    y=array([[1.2]], shape=(1, 1)),
                    heading=array([[1.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([1]),
            ),
            (  # Multiple well-separated obstacles persist
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.3, 10.1], shape=(K := 2,)),
                    y=array([0.1, 10.2], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([3, 7]),
            ),
            (  # New obstacle appears (one matches, one new)
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=10,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.3, 50.0], shape=(K := 2,)),
                    y=array([0.1, 50.0], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([3]),
                expected := data.obstacle_ids([3, 10]),
            ),
            (  # Obstacle disappears
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([10.1], shape=(K := 1,)),
                    y=array([10.2], shape=(K,)),
                    heading=array([0.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([7]),
            ),
            (  # Input order: swapped observation order, IDs follow obstacles
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([10.1, 0.3], shape=(K := 2,)),  # swapped vs history
                    y=array([10.2, 0.1], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([7, 3]),
            ),
            (  # Empty observations
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([], shape=(K := 0,)),
                    y=array([], shape=(K,)),
                    heading=array([], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.2, 10.0]], shape=(1, 2)),
                    y=array([[0.0, 10.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([]),
            ),
            (  # History padded with NaN columns (K_history > K_ids)
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=0.5,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.3, 10.1], shape=(K := 2,)),
                    y=array([0.1, 10.2], shape=(K,)),
                    heading=array([0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    # 4 columns, but only 2 are valid (rest are NaN padding)
                    x=array([[0.2, 10.0, np.nan, np.nan]], shape=(1, 4)),
                    y=array([[0.0, 10.0, np.nan, np.nan]], shape=(1, 4)),
                    heading=array([[0.0, 0.0, np.nan, np.nan]], shape=(1, 4)),
                ),
                # Only 2 IDs (matching the valid columns)
                ids := data.obstacle_ids([3, 7]),
                expected := data.obstacle_ids([3, 7]),
            ),
            (  # Far obstacle disappears, new far obstacle appears - nearby IDs should NOT swap
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    cutoff=10.0,  # Large enough for nearby obstacles
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.1, 5.1, -50.0], shape=(K := 3,)),  # At t+1: A, B, D
                    y=array([0.1, 0.1, 0.0], shape=(K,)),
                    heading=array([0.0, 0.0, 0.0], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0, 5.0, 50.0]], shape=(1, 3)),  # At t: A, B, C
                    y=array([[0.0, 0.0, 0.0]], shape=(1, 3)),
                    heading=array([[0.0, 0.0, 0.0]], shape=(1, 3)),
                ),
                ids := data.obstacle_ids([3, 7, 9]),  # IDs for A, B, C
                expected := data.obstacle_ids([3, 7, 1]),  # A->3, B->7, D->NEW
            ),
            (  # Obstacle disappears next to another obstacle moving in the opposite direction.
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    # In this case, need to compare the heading of the obstacles as well.
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 2,
                    start_id=100,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([10.0], shape=(K := 1,)),
                    y=array([3.5], shape=(K,)),
                    heading=array([np.pi], shape=(K,)),  # Moving left
                ),
                history := data.obstacle_2d_poses(
                    x=array([[10.0]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),  # Moving right
                ),
                ids := data.obstacle_ids([42]),
                expected := data.obstacle_ids([100]),
            ),
            (  # Same position, similar heading ⇒ same obstacle
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 4,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([1.0], shape=(K := 1,)),
                    y=array([0.0], shape=(K,)),
                    heading=array([0.1], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[1.0]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([5]),
            ),
            (  # Same position, heading difference just within cutoff ⇒ same obstacle
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 2,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([1.0], shape=(K := 1,)),
                    y=array([0.0], shape=(K,)),
                    heading=array([np.pi / 2 - 0.1], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[1.0]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([5]),
            ),
            (  # Same position, heading beyond cutoff ⇒ different obstacle
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 4,
                    start_id=10,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([1.0], shape=(K := 1,)),
                    y=array([0.0], shape=(K,)),
                    heading=array([np.pi], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[1.0]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([10]),
            ),
            (  # Heading -π and +π are the same ⇒ same obstacle
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 4,
                    start_id=1,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([1.0], shape=(K := 1,)),
                    y=array([0.0], shape=(K,)),
                    heading=array([np.pi - 0.1], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[1.0]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[-np.pi + 0.1]], shape=(1, 1)),
                ),
                ids := data.obstacle_ids([5]),
                expected := data.obstacle_ids([5]),
            ),
            (  # Two obstacles, only one has matching heading
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 4,
                    start_id=20,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([1.0, 2.0], shape=(K := 2,)),
                    y=array([0.0, 0.0], shape=(K,)),
                    heading=array([0.0, np.pi], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[1.0, 2.0]], shape=(1, 2)),
                    y=array([[0.0, 0.0]], shape=(1, 2)),
                    heading=array([[0.0, 0.0]], shape=(1, 2)),
                ),
                ids := data.obstacle_ids([5, 6]),
                expected := data.obstacle_ids([5, 20]),
            ),
            (  # Two obstacles, padded to three.
                assignment := id_assignment.hungarian(
                    position_extractor=position_extractor(),
                    orientation_extractor=heading_extractor(),
                    cutoff=5.0,
                    orientation_cutoff=np.pi / 4,
                    start_id=20,
                ),
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([1.0, 2.0], shape=(K := 2,)),
                    y=array([0.0, 0.0], shape=(K,)),
                    heading=array([0.0, np.pi], shape=(K,)),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[1.0, 2.0, np.nan]], shape=(T := 1, K_h := 3)),
                    y=array([[0.0, 0.0, np.nan]], shape=(T, K_h)),
                    heading=array([[0.0, 0.0, np.nan]], shape=(T, K_h)),
                ),
                ids := data.obstacle_ids([5, 6]),
                expected := data.obstacle_ids([5, 20]),
            ),
        ]

    @mark.parametrize(
        ["assignment", "states", "history", "ids", "expected"],
        [
            *cases(
                id_assignment=obstacles.numpy.id_assignment,
                position_extractor=NumPyObstaclePositionExtractor,
                heading_extractor=NumPyObstacleHeadingExtractor,
                data=data.numpy,
            ),
            *cases(
                id_assignment=obstacles.jax.id_assignment,
                position_extractor=JaxObstaclePositionExtractor,
                heading_extractor=JaxObstacleHeadingExtractor,
                data=data.jax,
            ),
        ],
    )
    def test[ObstacleStatesForTimeStepT, IdT: ObstacleIds, HistoryT](
        self,
        assignment: ObstacleIdAssignment[ObstacleStatesForTimeStepT, IdT, HistoryT],
        states: ObstacleStatesForTimeStepT,
        history: HistoryT,
        ids: IdT,
        expected: IdT,
    ) -> None:
        assert np.allclose(assignment(states, history=history, ids=ids), expected)
