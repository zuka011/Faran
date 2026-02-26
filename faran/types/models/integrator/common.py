from typing import Protocol

from faran.types.array import Array, DataType

from jaxtyping import Float


class IntegratorState(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, " D_x"]:
        """Returns the state as a NumPy array."""
        ...

    @property
    def dimension(self) -> int:
        """Returns the dimension of the state."""
        ...


class IntegratorStateSequence(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x"]:
        """Returns the state sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the state sequence."""
        ...

    @property
    def dimension(self) -> int:
        """State dimension."""
        ...


class IntegratorStateBatch(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x M"]:
        """Returns the states as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the state batch."""
        ...

    @property
    def dimension(self) -> int:
        """State dimension."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts in the batch."""
        ...


class IntegratorControlInputSequence(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u"]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> int:
        """Control input dimension."""
        ...


class IntegratorControlInputBatch(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u M"]:
        """Returns the control inputs as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """Time horizon of the control input batch."""
        ...

    @property
    def dimension(self) -> int:
        """Control input dimension."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts in the batch."""
        ...
