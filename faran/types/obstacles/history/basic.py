from typing import Protocol

from faran.types.array import Array
from faran.types.obstacles.history.common import (
    ObstaclePositionsForTimeStep,
    ObstaclePositions,
    ObstacleOrientationsForTimeStep,
    ObstacleOrientations,
    ObstaclePositionExtractor,
    ObstacleOrientationExtractor,
)

from jaxtyping import Float


class NumPyObstaclePositionsForTimeStep(ObstaclePositionsForTimeStep, Protocol):
    @property
    def array(self) -> Float[Array, "D_p K"]:
        """Returns the obstacle positions for a single time step as a NumPy array."""
        ...


class NumPyObstaclePositions(ObstaclePositions, Protocol):
    @property
    def array(self) -> Float[Array, "T D_p K"]:
        """Returns the obstacle positions as a NumPy array."""
        ...


class NumPyObstacleOrientationsForTimeStep(ObstacleOrientationsForTimeStep, Protocol):
    @property
    def array(self) -> Float[Array, "D_o K"]:
        """Returns the obstacle orientations for a single time step as a NumPy array."""
        ...


class NumPyObstacleOrientations(ObstacleOrientations, Protocol):
    @property
    def array(self) -> Float[Array, "T D_o K"]:
        """Returns the obstacle orientations as a NumPy array."""
        ...


class NumPyObstaclePositionExtractor[
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


class NumPyObstacleOrientationExtractor[
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
