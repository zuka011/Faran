from typing import Sequence

from faran.types import NumPyCosts, CostSumFunction

import numpy as np


class NumPyCostSumFunction[CostsT: NumPyCosts](CostSumFunction[CostsT]):
    """Sums multiple cost arrays element-wise into a single cost."""

    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        return initial.similar(
            array=np.sum([it.array for it in costs], axis=0) + initial.array
        )
