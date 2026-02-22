from typing import Protocol

from faran.types.array import DataType

from numtypes import Array, Dims


class ObstaclePositionsForTimeStep[D_p: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_p, K]]:
        """Returns the obstacle positions for a single time step as a NumPy array."""
        ...

    @property
    def dimension(self) -> D_p:
        """The dimension of the obstacle positions."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstaclePositions[T: int, D_p: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_p, K]]:
        """Returns the obstacle positions as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon of the obstacle positions."""
        ...

    @property
    def dimension(self) -> D_p:
        """The dimension of the obstacle positions."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleOrientationsForTimeStep[D_o: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        """Returns the obstacle orientations for a single time step as a NumPy array."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of the obstacle orientations."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleOrientations[T: int, D_o: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle orientations as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon of the obstacle orientations."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of the obstacle orientations."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstaclePositionExtractor[
    ObstacleStatesForTimeStepT,
    ObstacleStatesT,
    PositionsForTimeStepT,
    PositionsT,
](Protocol):
    def of_states_for_time_step(
        self, states: ObstacleStatesForTimeStepT, /
    ) -> PositionsForTimeStepT:
        """Extracts the positions from the given obstacle states for a single time step."""
        ...

    def of_states(self, states: ObstacleStatesT, /) -> PositionsT:
        """Extracts the positions from the given obstacle states."""
        ...


class ObstacleOrientationExtractor[
    ObstacleStatesForTimeStepT,
    ObstacleStatesT,
    OrientationsForTimeStepT,
    OrientationsT,
](Protocol):
    def of_states_for_time_step(
        self, states: ObstacleStatesForTimeStepT, /
    ) -> OrientationsForTimeStepT:
        """Extracts the orientations from the given obstacle states for a single time step."""
        ...

    def of_states(self, states: ObstacleStatesT, /) -> OrientationsT:
        """Extracts the orientations from the given obstacle states."""
        ...
