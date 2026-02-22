from typing import Protocol

from faran.types.obstacles.history.common import (
    ObstaclePositionsForTimeStep,
    ObstaclePositions,
    ObstacleOrientationsForTimeStep,
    ObstacleOrientations,
    ObstaclePositionExtractor,
    ObstacleOrientationExtractor,
)

from numtypes import Array, Dims


class NumPyObstaclePositionsForTimeStep[D_p: int, K: int](
    ObstaclePositionsForTimeStep[D_p, K], Protocol
):
    @property
    def array(self) -> Array[Dims[D_p, K]]:
        """Returns the obstacle positions for a single time step as a NumPy array."""
        ...


class NumPyObstaclePositions[T: int, D_p: int, K: int](
    ObstaclePositions[T, D_p, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_p, K]]:
        """Returns the obstacle positions as a NumPy array."""
        ...


class NumPyObstacleOrientationsForTimeStep[D_o: int, K: int](
    ObstacleOrientationsForTimeStep[D_o, K], Protocol
):
    @property
    def array(self) -> Array[Dims[D_o, K]]:
        """Returns the obstacle orientations for a single time step as a NumPy array."""
        ...


class NumPyObstacleOrientations[T: int, D_o: int, K: int](
    ObstacleOrientations[T, D_o, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
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
