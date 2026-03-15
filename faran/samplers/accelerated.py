from typing import Sequence

from faran.types import Array

from jaxtyping import Float, Array as JaxArray

import numpy as np
import jax.numpy as jnp


type StandardDeviationDescription = (
    Float[Array, " D_u"] | Float[JaxArray, " D_u"] | Sequence[float]
)


class standardize:
    @staticmethod
    def std(
        standard_deviation: StandardDeviationDescription,
    ) -> Float[JaxArray, " D_u"]:
        if isinstance(standard_deviation, (Sequence, np.ndarray)):
            return jnp.asarray(standard_deviation)

        return standard_deviation
