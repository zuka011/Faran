from typing import Sequence

from faran.types import Array

from jaxtyping import Float

import numpy as np


type StandardDeviationDescription = Float[Array, " D_u"] | Sequence[float]


class standardize:
    @staticmethod
    def std(standard_deviation: StandardDeviationDescription) -> Float[Array, " D_u"]:
        if isinstance(standard_deviation, Sequence):
            return np.asarray(standard_deviation)

        return standard_deviation
