from typing import cast
from dataclasses import dataclass

from faran.types import NumPyObstacleStateSampler
from faran.obstacles.basic import NumPySampledObstacle2dPoses, NumPyObstacle2dPoses

from riskit import distribution

import numpy as np


type Rng = np.random.Generator


@dataclass(frozen=True)
class NumPyGaussianObstacle2dPoseSampler(
    NumPyObstacleStateSampler[NumPyObstacle2dPoses, NumPySampledObstacle2dPoses]
):
    """Samples obstacle poses from a Gaussian distribution parameterized by predicted covariances."""

    rng: Rng

    @staticmethod
    def create(*, seed: int = 42) -> "NumPyGaussianObstacle2dPoseSampler":
        return NumPyGaussianObstacle2dPoseSampler(rng=np.random.default_rng(seed))

    def __call__(
        self, states: NumPyObstacle2dPoses, *, count: int
    ) -> NumPySampledObstacle2dPoses:
        if states.count == 0:
            return cast(NumPySampledObstacle2dPoses, states.single())

        if (covariance := states.covariance()) is None:
            assert count == 1, (
                "It's pointless to take multiple samples, when covariance information is not available."
            )
            return cast(NumPySampledObstacle2dPoses, states.single())

        T, D_O, _, K = covariance.shape

        mean = np.stack([states.x(), states.y(), states.heading()], axis=1)
        flat_covariance = covariance.transpose(0, 3, 1, 2).reshape(-1, D_O, D_O)
        flat_mean = mean.transpose(0, 2, 1).reshape(-1, D_O)

        samples = distribution.numpy.gaussian(
            mean=flat_mean, covariance=flat_covariance, rng=self.rng
        ).sample(count=count)

        samples = samples.reshape(T, K, D_O, count).transpose(0, 2, 1, 3)

        return states.sampled(
            x=samples[:, 0, :, :], y=samples[:, 1, :, :], heading=samples[:, 2, :, :]
        )
