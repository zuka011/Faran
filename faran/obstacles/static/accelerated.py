from typing import Self
from dataclasses import dataclass

from faran.types import Array, jaxtyped, JaxObstacleSimulator
from faran.obstacles.accelerated import JaxObstacle2dPosesForTimeStep

from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@jaxtyped
@dataclass(frozen=True)
class JaxStaticObstacleSimulator(JaxObstacleSimulator[JaxObstacle2dPosesForTimeStep]):
    """Simulates stationary obstacles by replicating fixed positions over the horizon."""

    positions: Float[JaxArray, "K 2"]
    headings: Float[JaxArray, " K"]

    @staticmethod
    def empty() -> "JaxStaticObstacleSimulator":
        positions = jnp.empty((0, 2))

        return JaxStaticObstacleSimulator.create(positions=positions)

    @staticmethod
    def create(
        *,
        positions: Float[Array, "K 2"] | Float[JaxArray, "K 2"],
        headings: Float[Array, " K"] | Float[JaxArray, " K"] | None = None,
    ) -> "JaxStaticObstacleSimulator":
        count = positions.shape[0]
        headings = headings if headings is not None else jnp.zeros(shape=(count,))

        return JaxStaticObstacleSimulator(
            positions=jnp.asarray(positions), headings=jnp.asarray(headings)
        )

    def with_time_step_size(self, time_step_size: float) -> Self:
        # Time step does not matter.
        return self

    def step(self) -> JaxObstacle2dPosesForTimeStep:
        return JaxObstacle2dPosesForTimeStep.create(
            x=self.positions[:, 0], y=self.positions[:, 1], heading=self.headings
        )

    @property
    def obstacle_count(self) -> int:
        return self.positions.shape[0]
