from typing import Protocol

from faran.types.array import Array
from faran.types.predictors import ObstacleStatesHistory

from jaxtyping import Float


class NumPyBicycleObstacleStatesHistory(ObstacleStatesHistory, Protocol):
    def x(self) -> Float[Array, "T K"]:
        """Returns the x positions of the obstacles over time."""
        ...

    def y(self) -> Float[Array, "T K"]:
        """Returns the y positions of the obstacles over time."""
        ...

    def heading(self) -> Float[Array, "T K"]:
        """Returns the headings of the obstacles over time."""
        ...

    @property
    def array(self) -> Float[Array, "T _ K"]:
        """Returns the obstacle history as a NumPy array."""
        ...
