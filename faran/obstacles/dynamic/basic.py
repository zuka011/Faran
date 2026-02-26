from typing import Self
from dataclasses import dataclass

from faran.types import Array, jaxtyped, NumPyObstacleSimulator
from faran.obstacles.basic import NumPyObstacle2dPosesForTimeStep

from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(kw_only=True)
class NumPyDynamicObstacleSimulator(
    NumPyObstacleSimulator[NumPyObstacle2dPosesForTimeStep]
):
    """Simulates obstacles moving with constant velocity over the prediction horizon."""

    last: NumPyObstacle2dPosesForTimeStep
    velocities: Float[Array, "K 2"]

    time_step: float | None = None

    @staticmethod
    def create(
        *, positions: Float[Array, "K 2"], velocities: Float[Array, "K 2"]
    ) -> "NumPyDynamicObstacleSimulator":
        headings = headings_from(velocities)

        return NumPyDynamicObstacleSimulator(
            last=NumPyObstacle2dPosesForTimeStep.create(
                x=positions[:, 0], y=positions[:, 1], heading=headings
            ),
            velocities=velocities,
        )

    def with_time_step_size(self, time_step_size: float) -> Self:
        return self.__class__(
            last=self.last, velocities=self.velocities, time_step=time_step_size
        )

    def step(self) -> NumPyObstacle2dPosesForTimeStep:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        x, y = step_obstacles(
            x=self.last.x(),
            y=self.last.y(),
            velocities=self.velocities,
            time_step=self.time_step,
        )

        self.last = NumPyObstacle2dPosesForTimeStep.create(
            x=x, y=y, heading=self.last.heading()
        )

        return self.last

    @property
    def obstacle_count(self) -> int:
        return self.last.count


def step_obstacles(
    *,
    x: Float[Array, " K"],
    y: Float[Array, " K"],
    velocities: Float[Array, "K 2"],
    time_step: float,
) -> tuple[Float[Array, " K"], Float[Array, " K"]]:
    new_x = x + velocities[:, 0] * time_step
    new_y = y + velocities[:, 1] * time_step

    return new_x, new_y


def headings_from(velocities: Float[Array, "K 2"]) -> Float[Array, " K"]:
    speed = np.linalg.norm(velocities, axis=1)
    heading = np.where(
        speed > 1e-6, np.arctan2(velocities[:, 1], velocities[:, 0]), 0.0
    )

    return heading
