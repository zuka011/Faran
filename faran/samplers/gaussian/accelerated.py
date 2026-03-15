from typing import Final
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    JaxControlInputBatchCreator,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxSampler,
)
from faran.samplers.accelerated import StandardDeviationDescription, standardize

from jaxtyping import Array as JaxArray, Float, PRNGKeyArray

import jax
import jax.random as jrandom


@dataclass(kw_only=True)
class JaxGaussianSampler[BatchT: JaxControlInputBatch](
    JaxSampler[JaxControlInputSequence, BatchT]
):
    """Perturbs a nominal control sequence with zero-mean Gaussian noise."""

    standard_deviation: Final[Float[JaxArray, " D_u"]]
    to_batch: Final[JaxControlInputBatchCreator[BatchT]]

    _rollout_count: Final[int]

    key: PRNGKeyArray

    @staticmethod
    def create[B: JaxControlInputBatch](
        *,
        standard_deviation: StandardDeviationDescription,
        rollout_count: int,
        to_batch: JaxControlInputBatchCreator,
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return JaxGaussianSampler(
            standard_deviation=standardize.std(standard_deviation),
            to_batch=to_batch,
            _rollout_count=rollout_count,
            key=key if key is not None else jrandom.key(seed or 0),
        )

    def sample(self, *, around: JaxControlInputSequence) -> BatchT:
        self.key, samples = sample(
            self.key,
            around=around.array,
            standard_deviation=self.standard_deviation,
            rollout_count=self.rollout_count,
        )

        return self.to_batch(array=samples)

    @property
    def rollout_count(self) -> int:
        return self._rollout_count


@jax.jit(static_argnames=("rollout_count",))
@jaxtyped
def sample(
    key: PRNGKeyArray,
    *,
    around: Float[JaxArray, "T D_u"],
    standard_deviation: Float[JaxArray, " D_u"],
    rollout_count: int,
) -> tuple[PRNGKeyArray, Float[JaxArray, "T D_u M"]]:
    time_horizon, control_dimension = around.shape

    key, subkey = jrandom.split(key)
    samples = around[..., None] + standard_deviation[None, :, None] * jrandom.normal(
        subkey, shape=(time_horizon, control_dimension, rollout_count)
    )

    return key, samples
