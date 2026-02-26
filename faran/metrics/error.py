from typing import Any
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
    StateSequence,
    ContouringCost,
    LagCost,
    SimulationData,
    Metric,
)
from faran.collectors import access

from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class MpccErrorMetricResult:
    """Results of the MPCC error metric, including contouring and lag errors."""

    contouring: Float[Array, " T"]
    lag: Float[Array, " T"]

    @cached_property
    def max_contouring(self) -> float:
        """Returns the maximum absolute contouring error detected."""
        return float(np.abs(self.contouring).max())

    @cached_property
    def max_lag(self) -> float:
        """Returns the maximum absolute lag error detected."""
        return float(np.abs(self.lag).max())


@dataclass(kw_only=True, frozen=True)
class MpccErrorMetric[StateBatchT](Metric[MpccErrorMetricResult]):
    """Metric evaluating contouring and lag errors in MPCC tracking."""

    contouring: ContouringCost[Any, StateBatchT]
    lag: LagCost[Any, StateBatchT]

    @staticmethod
    def create(
        *,
        contouring: ContouringCost[Any, StateBatchT],
        lag: LagCost[Any, StateBatchT],
    ) -> "MpccErrorMetric[StateBatchT]":
        return MpccErrorMetric(contouring=contouring, lag=lag)

    def compute(self, data: SimulationData) -> MpccErrorMetricResult:
        states = data(access.states.assume(StateSequence[StateBatchT]).require())
        state_batch = states.batched()

        contouring_error = self.contouring.error(states=state_batch)
        lag_error = self.lag.error(states=state_batch)

        return MpccErrorMetricResult(
            contouring=np.asarray(contouring_error)[..., 0],
            lag=np.asarray(lag_error)[..., 0],
        )

    @property
    def name(self) -> str:
        return "mpcc-error"
