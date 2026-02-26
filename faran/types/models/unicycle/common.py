from typing import Protocol, Final

from faran.types.array import Array, DataType

from jaxtyping import Float
from numtypes import D

UNICYCLE_D_X: Final = 3
UNICYCLE_D_U: Final = 2
UNICYCLE_D_O: Final = 3

type UnicycleD_x = D[3]
"""State dimension of the unicycle model, consisting of (x position, y position, heading)."""

type UnicycleD_u = D[2]
"""Control input dimension of the unicycle model, consisting of (linear velocity, angular velocity)."""

type UnicycleD_o = D[3]
"""Obstacle state dimension of the unicycle model, consisting of (x position, y position, heading)."""


class UnicycleState(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"{UNICYCLE_D_X}"]:
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
    def dimension(self) -> UnicycleD_x:
        """State dimension."""
        ...


class UnicycleStateSequence(Protocol):
    def step(self, index: int) -> UnicycleState:
        """Returns the state at the given time step index."""
        ...


class UnicycleStateBatch(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"T {UNICYCLE_D_X} M"]:
        """Returns the states as a NumPy array."""
        ...

    def heading(self) -> Float[Array, "T M"]:
        """Returns the headings (orientations) of the states in the batch."""
        ...

    def rollout(self, index: int) -> UnicycleStateSequence:
        """Returns a single rollout from the batch as a state sequence."""
        ...

    @property
    def positions(self) -> "UnicyclePositions":
        """Returns the positions of the states in the batch."""
        ...


class UnicyclePositions(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 M"]:
        """Returns the positions as a NumPy array."""
        ...

    def x(self) -> Float[Array, "T M"]:
        """Returns the x positions."""
        ...

    def y(self) -> Float[Array, "T M"]:
        """Returns the y positions."""
        ...


class UnicycleControlInputSequence(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"T {UNICYCLE_D_U}"]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> UnicycleD_u:
        """Control input dimension."""
        ...


class UnicycleControlInputBatch(Protocol):
    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, f"T {UNICYCLE_D_U} M"]:
        """Returns the control inputs as a NumPy array."""
        ...

    def linear_velocity(self) -> Float[Array, "T M"]:
        """Returns the linear velocities over time for each rollout."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts in the batch."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control inputs."""
        ...
