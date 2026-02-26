from typing import Protocol

from faran.types.obstacles.history.common import (
    ObstaclePositionsForTimeStep,
    ObstaclePositions,
    ObstaclePositionExtractor,
    ObstacleOrientationsForTimeStep,
    ObstacleOrientations,
    ObstacleOrientationExtractor,
)

from jaxtyping import Array as JaxArray, Float


class JaxObstaclePositionsForTimeStep(ObstaclePositionsForTimeStep, Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_p K"]:
        """Returns the obstacle positions for a single time step as a JAX array."""
        ...


class JaxObstaclePositions(ObstaclePositions, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_p K"]:
        """Returns the obstacle positions as a JAX array."""
        ...


class JaxObstacleOrientationsForTimeStep(ObstacleOrientationsForTimeStep, Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        """Returns the obstacle orientations for a single time step as a JAX array."""
        ...


class JaxObstacleOrientations(ObstacleOrientations, Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle orientations as a JAX array."""
        ...


class JaxObstaclePositionExtractor[
    ObstacleStatesForTimeStepT,
    ObstacleStatesT,
    PositionsForTimeStepT,
    PositionsT,
](
    ObstaclePositionExtractor[
        ObstacleStatesForTimeStepT, ObstacleStatesT, PositionsForTimeStepT, PositionsT
    ],
    Protocol,
): ...


class JaxObstacleOrientationExtractor[
    ObstacleStatesForTimeStepT,
    ObstacleStatesT,
    OrientationsForTimeStepT,
    OrientationsT,
](
    ObstacleOrientationExtractor[
        ObstacleStatesForTimeStepT,
        ObstacleStatesT,
        OrientationsForTimeStepT,
        OrientationsT,
    ],
    Protocol,
): ...
