from typing import Protocol, Final

from faran.types.array import Array, DataType

from jaxtyping import Float
from numtypes import D

BICYCLE_D_X: Final = 4
BICYCLE_D_U: Final = 2
BICYCLE_D_O: Final = 4

type BicycleD_x = D[4]
"""State dimension of the bicycle model, consisting of (x position, y position, heading, speed)."""

type BicycleD_u = D[2]
"""Control input dimension of the bicycle model, consisting of (acceleration, steering angle)."""

type BicycleD_o = D[4]
"""Obstacle state dimension of the bicycle model, consisting of (x position, y position, heading, speed)."""


class BicycleState(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"{BICYCLE_D_X}"]:
        """Returns the state as a NumPy array."""
        ...

    @property
    def x(self) -> float:
        """X position of the agent."""
        ...

    @property
    def y(self) -> float:
        """Y position of the agent."""
        ...

    @property
    def heading(self) -> float:
        """Orientation of the agent."""
        ...

    @property
    def speed(self) -> float:
        """Velocity of the agent."""
        ...

    @property
    def dimension(self) -> BicycleD_x:
        """State dimension."""
        ...


class BicycleStateSequence(Protocol):
    def step(self, index: int) -> BicycleState:
        """Returns the state at the given time step index."""
        ...


class BicycleStateBatch(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"T {BICYCLE_D_X} M"]:
        """Returns the states as a NumPy array."""
        ...

    def heading(self) -> Float[Array, "T M"]:
        """Returns the headings (orientations) of the states in the batch."""
        ...

    def speed(self) -> Float[Array, "T M"]:
        """Returns the speeds of the states in the batch."""
        ...

    def rollout(self, index: int) -> BicycleStateSequence:
        """Returns a single rollout from the batch as a state sequence."""
        ...

    @property
    def positions(self) -> "BicyclePositions":
        """Returns the positions of the states in the batch."""
        ...


class BicyclePositions(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 M"]:
        """Returns the positions as a NumPy array."""
        ...

    def x(self) -> Float[Array, "T M"]:
        """Returns the x positions."""
        ...

    def y(self) -> Float[Array, "T M"]:
        """Returns the y positions."""
        ...


class BicycleControlInputSequence(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"T {BICYCLE_D_U}"]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> BicycleD_u:
        """Control input dimension."""
        ...


class BicycleControlInputBatch(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"T {BICYCLE_D_U} M"]:
        """Returns the control inputs as a NumPy array."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts in the batch."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control inputs."""
        ...
