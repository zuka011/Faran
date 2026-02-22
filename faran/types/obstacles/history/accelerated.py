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


class JaxObstaclePositionsForTimeStep[D_p: int, K: int](
    ObstaclePositionsForTimeStep[D_p, K], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "D_p K"]:
        """Returns the obstacle positions for a single time step as a JAX array."""
        ...


class JaxObstaclePositions[T: int, D_p: int, K: int](
    ObstaclePositions[T, D_p, K], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_p K"]:
        """Returns the obstacle positions as a JAX array."""
        ...


class JaxObstacleOrientationsForTimeStep[D_o: int, K: int](
    ObstacleOrientationsForTimeStep[D_o, K], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        """Returns the obstacle orientations for a single time step as a JAX array."""
        ...


class JaxObstacleOrientations[T: int, D_o: int, K: int](
    ObstacleOrientations[T, D_o, K], Protocol
):
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
