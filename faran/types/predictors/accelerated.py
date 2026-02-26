from typing import Protocol

from faran.types.predictors.common import ObstacleStatesHistory

from jaxtyping import Float, Array as JaxArray


class JaxObstacleStatesHistory[ObstacleStatesForTimeStepT](
    ObstacleStatesHistory[ObstacleStatesForTimeStepT], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle state history as a JAX array."""
        ...
