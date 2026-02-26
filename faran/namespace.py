from typing import Final, Any

from faran.types import (
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyCostFunction,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxCosts,
    JaxCostFunction,
    State as State,
    StateSequence as StateSequence,
    StateBatch as StateBatch,
    ControlInputSequence as ControlInputSequence,
    ControlInputBatch as ControlInputBatch,
    Costs as Costs,
    CostFunction as CostFunction,
    BICYCLE_D_X,
    BICYCLE_D_U,
    BicycleD_x,
    BicycleD_u,
    BicycleState,
    BicycleStateSequence,
    BicycleStateBatch,
    BicyclePositions,
    BicycleControlInputSequence,
    BicycleControlInputBatch,
    UNICYCLE_D_X,
    UNICYCLE_D_U,
    UnicycleD_x,
    UnicycleD_u,
    UnicycleState,
    UnicycleStateSequence,
    UnicycleStateBatch,
    UnicyclePositions,
    UnicycleControlInputSequence,
    UnicycleControlInputBatch,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyHeadings,
    NumPyLateralPositions,
    NumPyLongitudinalPositions,
    NumPyNormals,
    NumPyRisk,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxHeadings,
    JaxLateralPositions,
    JaxLongitudinalPositions,
    JaxNormals,
    JaxRisk,
    PoseD_o as PoseD_o_,
    POSE_D_O as POSE_D_O_,
    NumPyPathParameterExtractor,
    NumPyPathVelocityExtractor,
    NumPyPositionExtractor,
    NumPyDistanceExtractor,
    NumPyRiskMetric,
    JaxPathParameterExtractor,
    JaxPathVelocityExtractor,
    JaxPositionExtractor,
    JaxDistanceExtractor,
    JaxRiskMetric,
    Error as Error,  # NOTE: Aliased to workaround ruff bug.
    Risk as Risk,
    RiskMetric as RiskMetric,
    ContouringCost as ContouringCost,
    NumPySampledObstacleStates,
    NumPySampledObstaclePositions,
    NumPySampledObstacleHeadings,
    NumPySampledObstaclePositionExtractor,
    NumPySampledObstacleHeadingExtractor,
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
    NumPyObstacleStateProvider,
    NumPyObstaclePositionExtractor,
    JaxSampledObstacleStates,
    JaxSampledObstaclePositions,
    JaxSampledObstacleHeadings,
    JaxSampledObstaclePositionExtractor,
    JaxSampledObstacleHeadingExtractor,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
    JaxObstacleStateProvider,
    JaxObstaclePositionExtractor,
    AugmentedState,
    AugmentedStateSequence,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    NumPyBoundaryDistance,
    NumPyBoundaryDistanceExtractor,
    JaxBoundaryDistance,
    JaxBoundaryDistanceExtractor,
)
from faran.models import (
    NumPyBicycleState,
    NumPyBicycleStateSequence,
    NumPyBicycleStateBatch,
    NumPyBicyclePositions,
    NumPyBicycleControlInputSequence,
    NumPyBicycleControlInputBatch,
    NumPyBicycleObstacleStates,
    NumPyBicycleObstacleInputs,
    NumPyBicycleObstacleStateSequences,
    NumPyIntegratorObstacleStateSequences,
    JaxBicycleState,
    JaxBicycleStateSequence,
    JaxBicycleStateBatch,
    JaxBicyclePositions,
    JaxBicycleControlInputSequence,
    JaxBicycleControlInputBatch,
    JaxBicycleObstacleStates,
    JaxBicycleObstacleInputs,
    JaxBicycleObstacleStateSequences,
    JaxIntegratorObstacleStateSequences,
    NumPyUnicycleState,
    NumPyUnicycleStateSequence,
    NumPyUnicycleStateBatch,
    NumPyUnicyclePositions,
    NumPyUnicycleControlInputSequence,
    NumPyUnicycleControlInputBatch,
    NumPyUnicycleObstacleStates,
    NumPyUnicycleObstacleInputs,
    NumPyUnicycleObstacleStateSequences,
    JaxUnicycleState,
    JaxUnicycleStateSequence,
    JaxUnicycleStateBatch,
    JaxUnicyclePositions,
    JaxUnicycleControlInputSequence,
    JaxUnicycleControlInputBatch,
    JaxUnicycleObstacleStates,
    JaxUnicycleObstacleInputs,
    JaxUnicycleObstacleStateSequences,
)
from faran.costs import (
    NumPyContouringCost,
    JaxContouringCost,
    NumPyDistance,
    JaxDistance,
)
from faran.obstacles import (
    NumPyObstacleIds,
    NumPySampledObstacle2dPoses,
    NumPyObstacle2dPoses,
    NumPyObstacle2dPosesForTimeStep,
    NumPyObstacle2dPositions,
    NumPyObstacle2dPositionsForTimeStep,
    NumPyObstacleHeadings,
    NumPyObstacleHeadingsForTimeStep,
    NumPyObstacleStatesRunningHistory,
    JaxObstacleIds,
    JaxSampledObstacle2dPoses,
    JaxObstacle2dPoses,
    JaxObstacle2dPosesForTimeStep,
    JaxObstacle2dPositions,
    JaxObstacle2dPositionsForTimeStep,
    JaxObstacleHeadings,
    JaxObstacleHeadingsForTimeStep,
    JaxObstacleStatesRunningHistory,
)
from faran.states import (
    NumPySimpleState,
    NumPySimpleStateSequence,
    NumPySimpleStateBatch,
    NumPySimpleControlInputSequence,
    NumPySimpleControlInputBatch,
    NumPySimpleCosts,
    NumPySimpleSampledObstacleStates,
    NumPySimpleObstacleStatesForTimeStep,
    NumPySimpleObstacleStates,
    JaxSimpleState,
    JaxSimpleStateSequence,
    JaxSimpleStateBatch,
    JaxSimpleControlInputSequence,
    JaxSimpleControlInputBatch,
    JaxSimpleCosts,
    JaxSimpleSampledObstacleStates,
    JaxSimpleObstacleStatesForTimeStep,
    JaxSimpleObstacleStates,
    NumPyAugmentedState,
    NumPyAugmentedStateSequence,
    NumPyAugmentedStateBatch,
    NumPyAugmentedControlInputSequence,
    NumPyAugmentedControlInputBatch,
    JaxAugmentedState,
    JaxAugmentedStateSequence,
    JaxAugmentedStateBatch,
    JaxAugmentedControlInputSequence,
    JaxAugmentedControlInputBatch,
)


class types:
    """Namespace of type aliases for states, controls, costs, and related domain types."""

    type State = State
    type StateBatch = StateBatch
    type ControlInputSequence = ControlInputSequence
    type ControlInputBatch = ControlInputBatch
    type Costs = Costs
    type CostFunction[InputBatchT, StateBatchT, CostsT] = CostFunction[
        InputBatchT, StateBatchT, CostsT
    ]
    type Error = Error
    type Risk = Risk

    type ContouringCost[InputBatchT, StateBatchT, ErrorT] = ContouringCost[
        InputBatchT, StateBatchT, ErrorT
    ]
    type RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT] = (
        RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT]
    )

    class obstacle:
        type PoseD_o = PoseD_o_

        POSE_D_O: Final = POSE_D_O_

    class bicycle:
        type D_x = BicycleD_x
        type D_u = BicycleD_u
        type State = BicycleState
        type StateSequence = BicycleStateSequence
        type StateBatch = BicycleStateBatch
        type Positions = BicyclePositions
        type ControlInputSequence = BicycleControlInputSequence
        type ControlInputBatch = BicycleControlInputBatch

        D_X: Final = BICYCLE_D_X
        D_U: Final = BICYCLE_D_U

    class unicycle:
        type D_x = UnicycleD_x
        type D_u = UnicycleD_u
        type State = UnicycleState
        type StateSequence = UnicycleStateSequence
        type StateBatch = UnicycleStateBatch
        type Positions = UnicyclePositions
        type ControlInputSequence = UnicycleControlInputSequence
        type ControlInputBatch = UnicycleControlInputBatch

        D_X: Final = UNICYCLE_D_X
        D_U: Final = UNICYCLE_D_U

    class augmented:
        type State[P, V] = AugmentedState[P, V]
        type StateSequence[P, V] = AugmentedStateSequence[P, V]
        type StateBatch[P, V] = AugmentedStateBatch[P, V]
        type ControlInputSequence[P, V] = AugmentedControlInputSequence[P, V]
        type ControlInputBatch[P, V] = AugmentedControlInputBatch[P, V]

        state: Final = AugmentedState
        state_sequence: Final = AugmentedStateSequence
        state_batch: Final = AugmentedStateBatch
        control_input_sequence: Final = AugmentedControlInputSequence
        control_input_batch: Final = AugmentedControlInputBatch

    class numpy:
        type State = NumPyState
        type StateSequence = NumPyStateSequence
        type StateBatch = NumPyStateBatch
        type ControlInputSequence = NumPyControlInputSequence
        type ControlInputBatch = NumPyControlInputBatch
        type Costs = NumPyCosts
        type PathParameters = NumPyPathParameters
        type ReferencePoints = NumPyReferencePoints
        type Positions = NumPyPositions
        type Headings = NumPyHeadings
        type LateralPositions = NumPyLateralPositions
        type LongitudinalPositions = NumPyLongitudinalPositions
        type ObstacleIds = NumPyObstacleIds
        type SampledObstacleStates = NumPySampledObstacleStates
        type SampledObstaclePositions = NumPySampledObstaclePositions
        type SampledObstacleHeadings = NumPySampledObstacleHeadings
        type SampledObstaclePositionExtractor[SampledStatesT] = (
            NumPySampledObstaclePositionExtractor[SampledStatesT]
        )
        type SampledObstacleHeadingExtractor[SampledStatesT] = (
            NumPySampledObstacleHeadingExtractor[SampledStatesT]
        )
        type ObstacleStates[SingleSampleT = Any, ObstacleStatesForTimeStepT = Any] = (
            NumPyObstacleStates[SingleSampleT, ObstacleStatesForTimeStepT]
        )
        type ObstacleStatesForTimeStep[ObstacleStatesT = Any] = (
            NumPyObstacleStatesForTimeStep[ObstacleStatesT]
        )
        type ObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ] = NumPyObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ]
        type SampledObstacle2dPoses = NumPySampledObstacle2dPoses
        type Obstacle2dPoses = NumPyObstacle2dPoses
        type Obstacle2dPosesForTimeStep = NumPyObstacle2dPosesForTimeStep
        type Obstacle2dPositions = NumPyObstacle2dPositions
        type Obstacle2dPositionsForTimeStep = NumPyObstacle2dPositionsForTimeStep
        type ObstacleHeadings = NumPyObstacleHeadings
        type ObstacleHeadingsForTimeStep = NumPyObstacleHeadingsForTimeStep
        type ObstacleStatesRunningHistory[
            StatesT,
            StatesForTimeStepT: NumPyObstacleStatesForTimeStep,
        ] = NumPyObstacleStatesRunningHistory[StatesT, StatesForTimeStepT]
        type Distance = NumPyDistance
        type BoundaryDistance = NumPyBoundaryDistance
        type Risk = NumPyRisk

        type CostFunction = NumPyCostFunction
        type PathParameterExtractor[StateBatchT] = NumPyPathParameterExtractor[
            StateBatchT
        ]
        type PathVelocityExtractor[InputBatchT] = NumPyPathVelocityExtractor[
            InputBatchT
        ]
        type PositionExtractor[StateBatchT] = NumPyPositionExtractor[StateBatchT]
        type DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT] = (
            NumPyDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT]
        )
        type BoundaryDistanceExtractor[StateBatchT, DistanceT] = (
            NumPyBoundaryDistanceExtractor[StateBatchT, DistanceT]
        )
        type RiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT] = (
            NumPyRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT]
        )
        type ContouringCost[StateBatchT] = NumPyContouringCost[StateBatchT]
        type ObstacleStateProvider[ObstacleStatesT] = NumPyObstacleStateProvider[
            ObstacleStatesT
        ]

        path_parameters: Final = NumPyPathParameters
        reference_points: Final = NumPyReferencePoints.create
        positions: Final = NumPyPositions.create
        headings: Final = NumPyHeadings.create
        lateral_positions: Final = NumPyLateralPositions.create
        longitudinal_positions: Final = NumPyLongitudinalPositions.create
        normals: Final = NumPyNormals.create
        distance: Final = NumPyDistance
        boundary_distance: Final = NumPyBoundaryDistance
        obstacle_ids: Final = NumPyObstacleIds
        obstacle_2d_poses: Final = NumPyObstacle2dPoses
        obstacle_2d_poses_for_time_step: Final = NumPyObstacle2dPosesForTimeStep
        obstacle_states_running_history: Final = NumPyObstacleStatesRunningHistory

        class simple:
            type State = NumPySimpleState
            type StateSequence = NumPySimpleStateSequence
            type StateBatch = NumPySimpleStateBatch
            type ControlInputSequence = NumPySimpleControlInputSequence
            type ControlInputBatch = NumPySimpleControlInputBatch
            type Costs = NumPySimpleCosts
            type SampledObstacleStates = NumPySimpleSampledObstacleStates
            type ObstacleStatesForTimeStep = NumPySimpleObstacleStatesForTimeStep
            type ObstacleStates = NumPySimpleObstacleStates

            state: Final = NumPySimpleState
            state_sequence: Final = NumPySimpleStateSequence
            state_batch: Final = NumPySimpleStateBatch
            control_input_sequence: Final = NumPySimpleControlInputSequence
            control_input_batch: Final = NumPySimpleControlInputBatch
            costs: Final = NumPySimpleCosts
            obstacle_states: Final = NumPySimpleObstacleStates

        class integrator:
            type ObstacleStateSequences = NumPyIntegratorObstacleStateSequences

        class bicycle:
            type State = NumPyBicycleState
            type StateSequence = NumPyBicycleStateSequence
            type StateBatch = NumPyBicycleStateBatch
            type Positions = NumPyBicyclePositions
            type ControlInputSequence = NumPyBicycleControlInputSequence
            type ControlInputBatch = NumPyBicycleControlInputBatch
            type ObstacleStates = NumPyBicycleObstacleStates
            type ObstacleInputs = NumPyBicycleObstacleInputs
            type ObstacleStateSequences = NumPyBicycleObstacleStateSequences

            state: Final = NumPyBicycleState
            state_sequence: Final = NumPyBicycleStateSequence
            state_batch: Final = NumPyBicycleStateBatch
            positions: Final = NumPyBicyclePositions
            control_input_sequence: Final = NumPyBicycleControlInputSequence
            control_input_batch: Final = NumPyBicycleControlInputBatch
            obstacle_states: Final = NumPyBicycleObstacleStates
            obstacle_inputs: Final = NumPyBicycleObstacleInputs
            obstacle_state_sequences: Final = NumPyBicycleObstacleStateSequences

        class unicycle:
            type State = NumPyUnicycleState
            type StateSequence = NumPyUnicycleStateSequence
            type StateBatch = NumPyUnicycleStateBatch
            type Positions = NumPyUnicyclePositions
            type ControlInputSequence = NumPyUnicycleControlInputSequence
            type ControlInputBatch = NumPyUnicycleControlInputBatch
            type ObstacleStates = NumPyUnicycleObstacleStates
            type ObstacleInputs = NumPyUnicycleObstacleInputs
            type ObstacleStateSequences = NumPyUnicycleObstacleStateSequences

            state: Final = NumPyUnicycleState
            state_sequence: Final = NumPyUnicycleStateSequence
            state_batch: Final = NumPyUnicycleStateBatch
            positions: Final = NumPyUnicyclePositions
            control_input_sequence: Final = NumPyUnicycleControlInputSequence
            control_input_batch: Final = NumPyUnicycleControlInputBatch
            obstacle_states: Final = NumPyUnicycleObstacleStates
            obstacle_inputs: Final = NumPyUnicycleObstacleInputs
            obstacle_state_sequences: Final = NumPyUnicycleObstacleStateSequences

        class augmented:
            type State[P: NumPyState, V: NumPyState] = NumPyAugmentedState[P, V]
            type StateSequence[P: NumPyStateSequence, V: NumPyStateSequence] = (
                NumPyAugmentedStateSequence[P, V]
            )
            type StateBatch[P: NumPyStateBatch, V: NumPyStateBatch] = (
                NumPyAugmentedStateBatch[P, V]
            )
            type ControlInputSequence[
                P: NumPyControlInputSequence,
                V: NumPyControlInputSequence,
            ] = NumPyAugmentedControlInputSequence[P, V]
            type ControlInputBatch[
                P: NumPyControlInputBatch,
                V: NumPyControlInputBatch,
            ] = NumPyAugmentedControlInputBatch[P, V]

            state: Final = NumPyAugmentedState
            state_sequence: Final = NumPyAugmentedStateSequence
            state_batch: Final = NumPyAugmentedStateBatch
            control_input_sequence: Final = NumPyAugmentedControlInputSequence
            control_input_batch: Final = NumPyAugmentedControlInputBatch

    class jax:
        type State = JaxState
        type StateSequence = JaxStateSequence
        type StateBatch = JaxStateBatch
        type ControlInputSequence = JaxControlInputSequence
        type ControlInputBatch = JaxControlInputBatch
        type Costs = JaxCosts
        type PathParameters = JaxPathParameters
        type ReferencePoints = JaxReferencePoints
        type Positions = JaxPositions
        type Headings = JaxHeadings
        type LateralPositions = JaxLateralPositions
        type LongitudinalPositions = JaxLongitudinalPositions
        type ObstacleIds = JaxObstacleIds
        type SampledObstacleStates = JaxSampledObstacleStates
        type SampledObstaclePositions = JaxSampledObstaclePositions
        type SampledObstacleHeadings = JaxSampledObstacleHeadings
        type SampledObstaclePositionExtractor[SampledStatesT] = (
            JaxSampledObstaclePositionExtractor[SampledStatesT]
        )
        type SampledObstacleHeadingExtractor[SampledStatesT] = (
            JaxSampledObstacleHeadingExtractor[SampledStatesT]
        )
        type ObstacleStates[SingleSampleT = Any, ObstacleStatesForTimeStepT = Any] = (
            JaxObstacleStates[SingleSampleT, ObstacleStatesForTimeStepT]
        )
        type ObstacleStatesForTimeStep[ObstacleStatesT = Any, NumPyT = Any] = (
            JaxObstacleStatesForTimeStep[ObstacleStatesT, NumPyT]
        )
        type ObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ] = JaxObstaclePositionExtractor[
            ObstacleStatesForTimeStepT,
            ObstacleStatesT,
            PositionsForTimeStepT,
            PositionsT,
        ]
        type SampledObstacle2dPoses = JaxSampledObstacle2dPoses
        type Obstacle2dPoses = JaxObstacle2dPoses
        type Obstacle2dPosesForTimeStep = JaxObstacle2dPosesForTimeStep
        type Obstacle2dPositions = JaxObstacle2dPositions
        type Obstacle2dPositionsForTimeStep = JaxObstacle2dPositionsForTimeStep
        type ObstacleHeadings = JaxObstacleHeadings
        type ObstacleHeadingsForTimeStep = JaxObstacleHeadingsForTimeStep
        type ObstacleStatesRunningHistory[
            StatesT: JaxObstacleStates,
            StatesForTimeStepT: JaxObstacleStatesForTimeStep,
        ] = JaxObstacleStatesRunningHistory[StatesT, StatesForTimeStepT]
        type Distance = JaxDistance
        type BoundaryDistance = JaxBoundaryDistance
        type Risk = JaxRisk

        type CostFunction = JaxCostFunction
        type PathParameterExtractor[StateBatchT] = JaxPathParameterExtractor[
            StateBatchT
        ]
        type PathVelocityExtractor[InputBatchT] = JaxPathVelocityExtractor[InputBatchT]
        type PositionExtractor[StateBatchT] = JaxPositionExtractor[StateBatchT]
        type DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT] = (
            JaxDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT]
        )
        type BoundaryDistanceExtractor[StateBatchT, DistanceT] = (
            JaxBoundaryDistanceExtractor[StateBatchT, DistanceT]
        )
        type RiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT] = (
            JaxRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT]
        )
        type ContouringCost[StateBatchT] = JaxContouringCost[StateBatchT]
        type ObstacleStateProvider[ObstacleStatesT] = JaxObstacleStateProvider[
            ObstacleStatesT
        ]

        path_parameters: Final = JaxPathParameters.create
        reference_points: Final = JaxReferencePoints.create
        positions: Final = JaxPositions.create
        headings: Final = JaxHeadings.create
        lateral_positions: Final = JaxLateralPositions.create
        longitudinal_positions: Final = JaxLongitudinalPositions.create
        normals: Final = JaxNormals.create
        distance: Final = JaxDistance
        boundary_distance: Final = JaxBoundaryDistance
        obstacle_ids: Final = JaxObstacleIds
        obstacle_2d_poses: Final = JaxObstacle2dPoses
        obstacle_2d_poses_for_time_step: Final = JaxObstacle2dPosesForTimeStep
        obstacle_states_running_history: Final = JaxObstacleStatesRunningHistory

        class simple:
            type State = JaxSimpleState
            type StateSequence = JaxSimpleStateSequence
            type StateBatch = JaxSimpleStateBatch
            type ControlInputSequence = JaxSimpleControlInputSequence
            type ControlInputBatch = JaxSimpleControlInputBatch
            type Costs = JaxSimpleCosts
            type SampledObstacleStates = JaxSimpleSampledObstacleStates
            type ObstacleStatesForTimeStep = JaxSimpleObstacleStatesForTimeStep
            type ObstacleStates = JaxSimpleObstacleStates

            state: Final = JaxSimpleState
            state_sequence: Final = JaxSimpleStateSequence
            state_batch: Final = JaxSimpleStateBatch
            control_input_sequence: Final = JaxSimpleControlInputSequence
            control_input_batch: Final = JaxSimpleControlInputBatch
            costs: Final = JaxSimpleCosts
            obstacle_states: Final = JaxSimpleObstacleStates

        class integrator:
            type ObstacleStateSequences = JaxIntegratorObstacleStateSequences

        class bicycle:
            type State = JaxBicycleState
            type StateSequence = JaxBicycleStateSequence
            type StateBatch = JaxBicycleStateBatch
            type Positions = JaxBicyclePositions
            type ControlInputSequence = JaxBicycleControlInputSequence
            type ControlInputBatch = JaxBicycleControlInputBatch
            type ObstacleStates = JaxBicycleObstacleStates
            type ObstacleInputs = JaxBicycleObstacleInputs
            type ObstacleStateSequences = JaxBicycleObstacleStateSequences

            state: Final = JaxBicycleState
            state_sequence: Final = JaxBicycleStateSequence
            state_batch: Final = JaxBicycleStateBatch
            positions: Final = JaxBicyclePositions
            control_input_sequence: Final = JaxBicycleControlInputSequence
            control_input_batch: Final = JaxBicycleControlInputBatch
            obstacle_states: Final = JaxBicycleObstacleStates
            obstacle_inputs: Final = JaxBicycleObstacleInputs
            obstacle_state_sequences: Final = JaxBicycleObstacleStateSequences

        class unicycle:
            type State = JaxUnicycleState
            type StateSequence = JaxUnicycleStateSequence
            type StateBatch = JaxUnicycleStateBatch
            type Positions = JaxUnicyclePositions
            type ControlInputSequence = JaxUnicycleControlInputSequence
            type ControlInputBatch = JaxUnicycleControlInputBatch
            type ObstacleStates = JaxUnicycleObstacleStates
            type ObstacleInputs = JaxUnicycleObstacleInputs
            type ObstacleStateSequences = JaxUnicycleObstacleStateSequences

            state: Final = JaxUnicycleState
            state_sequence: Final = JaxUnicycleStateSequence
            state_batch: Final = JaxUnicycleStateBatch
            positions: Final = JaxUnicyclePositions
            control_input_sequence: Final = JaxUnicycleControlInputSequence
            control_input_batch: Final = JaxUnicycleControlInputBatch
            obstacle_states: Final = JaxUnicycleObstacleStates
            obstacle_inputs: Final = JaxUnicycleObstacleInputs
            obstacle_state_sequences: Final = JaxUnicycleObstacleStateSequences

        class augmented:
            type State[P: JaxState, V: JaxState] = JaxAugmentedState[P, V]
            type StateSequence[P: JaxStateSequence, V: JaxStateSequence] = (
                JaxAugmentedStateSequence[P, V]
            )
            type StateBatch[P: JaxStateBatch, V: JaxStateBatch] = (
                JaxAugmentedStateBatch[P, V]
            )
            type ControlInputSequence[
                P: JaxControlInputSequence,
                V: JaxControlInputSequence,
            ] = JaxAugmentedControlInputSequence[P, V]
            type ControlInputBatch[P: JaxControlInputBatch, V: JaxControlInputBatch] = (
                JaxAugmentedControlInputBatch[P, V]
            )

            state: Final = JaxAugmentedState
            state_sequence: Final = JaxAugmentedStateSequence
            state_batch: Final = JaxAugmentedStateBatch
            control_input_sequence: Final = JaxAugmentedControlInputSequence
            control_input_batch: Final = JaxAugmentedControlInputBatch


class classes:
    """Namespace of concrete protocol classes for states, controls, costs, and related types."""

    State: Final = State
    StateSequence: Final = StateSequence
    StateBatch: Final = StateBatch
    ControlInputSequence: Final = ControlInputSequence
    ControlInputBatch: Final = ControlInputBatch
    Costs: Final = Costs
    CostFunction: Final = CostFunction
    Error: Final = Error
    Risk: Final = Risk
    RiskMetric: Final = RiskMetric
    ContouringCost: Final = ContouringCost

    class bicycle:
        State: Final = BicycleState
        StateSequence: Final = BicycleStateSequence
        StateBatch: Final = BicycleStateBatch
        Positions: Final = BicyclePositions
        ControlInputSequence: Final = BicycleControlInputSequence
        ControlInputBatch: Final = BicycleControlInputBatch

    class unicycle:
        State: Final = UnicycleState
        StateSequence: Final = UnicycleStateSequence
        StateBatch: Final = UnicycleStateBatch
        Positions: Final = UnicyclePositions
        ControlInputSequence: Final = UnicycleControlInputSequence
        ControlInputBatch: Final = UnicycleControlInputBatch

    class augmented:
        State: Final = AugmentedState
        StateSequence: Final = AugmentedStateSequence
        StateBatch: Final = AugmentedStateBatch
        ControlInputSequence: Final = AugmentedControlInputSequence
        ControlInputBatch: Final = AugmentedControlInputBatch

    class numpy:
        State: Final = NumPyState
        StateSequence: Final = NumPyStateSequence
        StateBatch: Final = NumPyStateBatch
        ControlInputSequence: Final = NumPyControlInputSequence
        ControlInputBatch: Final = NumPyControlInputBatch
        Costs: Final = NumPyCosts
        PathParameters: Final = NumPyPathParameters
        ReferencePoints: Final = NumPyReferencePoints
        Positions: Final = NumPyPositions
        Headings: Final = NumPyHeadings
        LateralPositions: Final = NumPyLateralPositions
        LongitudinalPositions: Final = NumPyLongitudinalPositions
        ObstacleIds: Final = NumPyObstacleIds
        ObstacleStates: Final = NumPyObstacleStates
        ObstacleStatesForTimeStep: Final = NumPyObstacleStatesForTimeStep
        SampledObstacleStates: Final = NumPySampledObstacleStates
        SampledObstaclePositions: Final = NumPySampledObstaclePositions
        SampledObstacleHeadings: Final = NumPySampledObstacleHeadings
        SampledObstaclePositionExtractor: Final = NumPySampledObstaclePositionExtractor
        SampledObstacleHeadingExtractor: Final = NumPySampledObstacleHeadingExtractor
        SampledObstacle2dPoses: Final = NumPySampledObstacle2dPoses
        Obstacle2dPoses: Final = NumPyObstacle2dPoses
        Obstacle2dPosesForTimeStep: Final = NumPyObstacle2dPosesForTimeStep
        Obstacle2dPositions: Final = NumPyObstacle2dPositions
        Obstacle2dPositionsForTimeStep: Final = NumPyObstacle2dPositionsForTimeStep
        ObstacleHeadings: Final = NumPyObstacleHeadings
        ObstacleHeadingsForTimeStep: Final = NumPyObstacleHeadingsForTimeStep
        ObstacleStatesRunningHistory: Final = NumPyObstacleStatesRunningHistory
        ObstaclePositionExtractor: Final = NumPyObstaclePositionExtractor
        Distance: Final = NumPyDistance
        BoundaryDistance: Final = NumPyBoundaryDistance
        Risk: Final = NumPyRisk
        CostFunction: Final = NumPyCostFunction
        PathParameterExtractor: Final = NumPyPathParameterExtractor
        PathVelocityExtractor: Final = NumPyPathVelocityExtractor
        PositionExtractor: Final = NumPyPositionExtractor
        DistanceExtractor: Final = NumPyDistanceExtractor
        BoundaryDistanceExtractor: Final = NumPyBoundaryDistanceExtractor
        RiskMetric: Final = NumPyRiskMetric
        ContouringCost: Final = NumPyContouringCost
        ObstacleStateProvider: Final = NumPyObstacleStateProvider

        class simple:
            State: Final = NumPySimpleState
            StateSequence: Final = NumPySimpleStateSequence
            StateBatch: Final = NumPySimpleStateBatch
            ControlInputSequence: Final = NumPySimpleControlInputSequence
            ControlInputBatch: Final = NumPySimpleControlInputBatch
            Costs: Final = NumPySimpleCosts
            SampledObstacleStates: Final = NumPySimpleSampledObstacleStates
            ObstacleStatesForTimeStep: Final = NumPySimpleObstacleStatesForTimeStep
            ObstacleStates: Final = NumPySimpleObstacleStates

        class integrator:
            ObstacleStateSequences: Final = NumPyIntegratorObstacleStateSequences

        class bicycle:
            State: Final = NumPyBicycleState
            StateSequence: Final = NumPyBicycleStateSequence
            StateBatch: Final = NumPyBicycleStateBatch
            Positions: Final = NumPyBicyclePositions
            ControlInputSequence: Final = NumPyBicycleControlInputSequence
            ControlInputBatch: Final = NumPyBicycleControlInputBatch
            ObstacleStateSequences: Final = NumPyBicycleObstacleStateSequences

        class unicycle:
            State: Final = NumPyUnicycleState
            StateSequence: Final = NumPyUnicycleStateSequence
            StateBatch: Final = NumPyUnicycleStateBatch
            Positions: Final = NumPyUnicyclePositions
            ControlInputSequence: Final = NumPyUnicycleControlInputSequence
            ControlInputBatch: Final = NumPyUnicycleControlInputBatch
            ObstacleStateSequences: Final = NumPyUnicycleObstacleStateSequences

        class augmented:
            State: Final = NumPyAugmentedState
            StateSequence: Final = NumPyAugmentedStateSequence
            StateBatch: Final = NumPyAugmentedStateBatch
            ControlInputSequence: Final = NumPyAugmentedControlInputSequence
            ControlInputBatch: Final = NumPyAugmentedControlInputBatch

    class jax:
        State: Final = JaxState
        StateSequence: Final = JaxStateSequence
        StateBatch: Final = JaxStateBatch
        ControlInputSequence: Final = JaxControlInputSequence
        ControlInputBatch: Final = JaxControlInputBatch
        Costs: Final = JaxCosts
        PathParameters: Final = JaxPathParameters
        ReferencePoints: Final = JaxReferencePoints
        Positions: Final = JaxPositions
        Headings: Final = JaxHeadings
        LateralPositions: Final = JaxLateralPositions
        LongitudinalPositions: Final = JaxLongitudinalPositions
        ObstacleIds: Final = JaxObstacleIds
        ObstacleStates: Final = JaxObstacleStates
        ObstacleStatesForTimeStep: Final = JaxObstacleStatesForTimeStep
        SampledObstacleStates: Final = JaxSampledObstacleStates
        SampledObstaclePositions: Final = JaxSampledObstaclePositions
        SampledObstacleHeadings: Final = JaxSampledObstacleHeadings
        SampledObstaclePositionExtractor: Final = JaxSampledObstaclePositionExtractor
        SampledObstacleHeadingExtractor: Final = JaxSampledObstacleHeadingExtractor
        SampledObstacle2dPoses: Final = JaxSampledObstacle2dPoses
        Obstacle2dPoses: Final = JaxObstacle2dPoses
        Obstacle2dPosesForTimeStep: Final = JaxObstacle2dPosesForTimeStep
        Obstacle2dPositions: Final = JaxObstacle2dPositions
        Obstacle2dPositionsForTimeStep: Final = JaxObstacle2dPositionsForTimeStep
        ObstacleHeadings: Final = JaxObstacleHeadings
        ObstacleHeadingsForTimeStep: Final = JaxObstacleHeadingsForTimeStep
        ObstacleStatesRunningHistory: Final = JaxObstacleStatesRunningHistory
        ObstaclePositionExtractor: Final = JaxObstaclePositionExtractor
        Distance: Final = JaxDistance
        BoundaryDistance: Final = JaxBoundaryDistance
        Risk: Final = JaxRisk
        CostFunction: Final = JaxCostFunction
        PathParameterExtractor: Final = JaxPathParameterExtractor
        PathVelocityExtractor: Final = JaxPathVelocityExtractor
        PositionExtractor: Final = JaxPositionExtractor
        DistanceExtractor: Final = JaxDistanceExtractor
        BoundaryDistanceExtractor: Final = JaxBoundaryDistanceExtractor
        RiskMetric: Final = JaxRiskMetric
        ContouringCost: Final = JaxContouringCost
        ObstacleStateProvider: Final = JaxObstacleStateProvider

        class simple:
            State: Final = JaxSimpleState
            StateSequence: Final = JaxSimpleStateSequence
            StateBatch: Final = JaxSimpleStateBatch
            ControlInputSequence: Final = JaxSimpleControlInputSequence
            ControlInputBatch: Final = JaxSimpleControlInputBatch
            Costs: Final = JaxSimpleCosts
            SampledObstacleStates: Final = JaxSimpleSampledObstacleStates
            ObstacleStatesForTimeStep: Final = JaxSimpleObstacleStatesForTimeStep
            ObstacleStates: Final = JaxSimpleObstacleStates

        class integrator:
            ObstacleStateSequences: Final = JaxIntegratorObstacleStateSequences

        class bicycle:
            State: Final = JaxBicycleState
            StateSequence: Final = JaxBicycleStateSequence
            StateBatch: Final = JaxBicycleStateBatch
            Positions: Final = JaxBicyclePositions
            ControlInputSequence: Final = JaxBicycleControlInputSequence
            ControlInputBatch: Final = JaxBicycleControlInputBatch
            ObstacleStateSequences: Final = JaxBicycleObstacleStateSequences

        class unicycle:
            State: Final = JaxUnicycleState
            StateSequence: Final = JaxUnicycleStateSequence
            StateBatch: Final = JaxUnicycleStateBatch
            Positions: Final = JaxUnicyclePositions
            ControlInputSequence: Final = JaxUnicycleControlInputSequence
            ControlInputBatch: Final = JaxUnicycleControlInputBatch
            ObstacleStateSequences: Final = JaxUnicycleObstacleStateSequences

        class augmented:
            State: Final = JaxAugmentedState
            StateSequence: Final = JaxAugmentedStateSequence
            StateBatch: Final = JaxAugmentedStateBatch
            ControlInputSequence: Final = JaxAugmentedControlInputSequence
            ControlInputBatch: Final = JaxAugmentedControlInputBatch
