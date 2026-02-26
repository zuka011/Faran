from typing import Final, overload
from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    Array,
    JaxControlInputBatchCreator,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxSampler,
)

from jaxtyping import Array as JaxArray, Float, PRNGKeyArray

import jax
import jax.random as jrandom
import jax.numpy as jnp


@dataclass(kw_only=True)
class JaxGaussianSampler[BatchT: JaxControlInputBatch](
    JaxSampler[JaxControlInputSequence, BatchT]
):
    """Perturbs a nominal control sequence with zero-mean Gaussian noise."""

    standard_deviation: Final[Float[JaxArray, " D_u"]]
    to_batch: Final[JaxControlInputBatchCreator[BatchT]]

    _control_dimension: Final[int]
    _rollout_count: Final[int]

    key: PRNGKeyArray

    @overload
    @staticmethod
    def create[B: JaxControlInputBatch](
        *,
        standard_deviation: Float[Array, " D_u"],
        rollout_count: int,
        to_batch: JaxControlInputBatchCreator,
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        ...

    @overload
    @staticmethod
    def create[B: JaxControlInputBatch](
        *,
        standard_deviation: Float[JaxArray, " D_u"],
        control_dimension: int | None = None,
        rollout_count: int,
        to_batch: JaxControlInputBatchCreator,
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        ...

    @staticmethod
    def create[B: JaxControlInputBatch](
        *,
        standard_deviation: Float[Array, " D_u"] | Float[JaxArray, " D_u"],
        control_dimension: int | None = None,
        rollout_count: int,
        to_batch: JaxControlInputBatchCreator,
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return JaxGaussianSampler(
            standard_deviation=jnp.asarray(standard_deviation),
            to_batch=to_batch,
            _control_dimension=(
                control_dimension
                if control_dimension is not None
                else standard_deviation.shape[0]
            ),
            _rollout_count=rollout_count,
            key=key if key is not None else jrandom.key(seed or 0),
        )

    def __post_init__(self) -> None:
        assert self.standard_deviation.shape[0] == self.control_dimension, (
            f"Expected standard deviation with shape ({self.control_dimension},), "
            f"but got {self.standard_deviation.shape}"
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
    def control_dimension(self) -> int:
        return self._control_dimension

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
