from typing import Protocol

from faran.types.array import Array
from faran.types.predictors.common import ObstacleStatesHistory

from jaxtyping import Float


class NumPyObstacleStatesHistory[ObstacleStatesForTimeStepT](
    ObstacleStatesHistory[ObstacleStatesForTimeStepT], Protocol
):
    @property
    def array(self) -> Float[Array, "T D_o K"]:
        """Returns the obstacle state history as a NumPy array."""
        ...
