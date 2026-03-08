from dataclasses import dataclass

from faran.types import Array, ObstacleStatesForTimeStep, ObstacleStateObserver

from jaxtyping import Float

import numpy as np


class ObstacleStateCreator[ObstacleStatesFormTimeStepT]:
    def __call__(self, array: Float[Array, "D_o K"], /) -> ObstacleStatesFormTimeStepT:
        """Wraps the specified array of observed obstacle states in an object of the
        appropriate type."""
        ...


@dataclass(kw_only=True, frozen=True)
class NoisyObstacleStateObserver[ObstacleStatesForTimeStepT: ObstacleStatesForTimeStep]:
    inner: ObstacleStateObserver[ObstacleStatesForTimeStepT]
    to_states: ObstacleStateCreator[ObstacleStatesForTimeStepT]
    sigma: Float[Array, " D_o"]
    rng: np.random.Generator

    @staticmethod
    def decorate[OS: ObstacleStatesForTimeStep](
        observer: ObstacleStateObserver[OS],
        *,
        to_states: ObstacleStateCreator[OS],
        sigma: Float[Array, " D_o"],
        seed: int = 0,
    ) -> "NoisyObstacleStateObserver[OS]":
        """Creates an observer that adds zero-mean Gaussian noise to the observed states before delegating to
        the underlying observer.

        Args:
            inner: The underlying observer to which the noisy observations will be delegated.
            to_states: An object to create the appropriate type of obstacle states from the noisy state arrays.
            sigma: The standard deviation of the Gaussian noise to be added to each component of the observed states.
            seed: The seed for the random number generator used to generate the noise.
        """
        return NoisyObstacleStateObserver(
            inner=observer,
            to_states=to_states,
            sigma=sigma,
            rng=np.random.default_rng(seed),
        )

    def observe(self, states: ObstacleStatesForTimeStepT) -> None:
        self.inner.observe(
            self.to_states(
                (array := np.asarray(states))
                + self.rng.normal(0, self.sigma[:, np.newaxis], array.shape)
            )
        )
