from typing import Protocol
from dataclasses import dataclass
from functools import cached_property

from faran.types.array import Array, jaxtyped
from faran.types.trajectories.common import (
    D_R,
    PathParameters,
    ReferencePoints,
    Positions,
    LateralPositions,
    LongitudinalPositions,
    Normals,
)

from jaxtyping import Float

import numpy as np


@jaxtyped
@dataclass(frozen=True)
class NumPyPathParameters(PathParameters):
    array: Float[Array, "T M"]

    def __array__(self) -> Float[Array, "T M"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]


class NumPyPositions(Positions, Protocol):
    @property
    def array(self) -> Float[Array, "T 2 M"]:
        """Returns the positions as a NumPy array."""
        ...


@jaxtyped
@dataclass(frozen=True)
class NumPySimplePositions(NumPyPositions):
    _x: Float[Array, "T M"]
    _y: Float[Array, "T M"]

    @staticmethod
    def create(
        *, x: Float[Array, "T M"], y: Float[Array, "T M"]
    ) -> "NumPySimplePositions":
        """Creates a NumPy positions instance from x and y coordinate arrays."""
        return NumPySimplePositions(_x=x, _y=y)

    def __array__(self) -> Float[Array, "T 2 M"]:
        return self.array

    def x(self) -> Float[Array, "T M"]:
        return self._x

    def y(self) -> Float[Array, "T M"]:
        return self._y

    @property
    def horizon(self) -> int:
        return self._x.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._x.shape[1]

    @property
    def array(self) -> Float[Array, "T 2 M"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "T 2 M"]:
        return np.stack([self._x, self._y], axis=1)


@jaxtyped
@dataclass(frozen=True)
class NumPyHeadings:
    _heading: Float[Array, "T M"]

    @staticmethod
    def create(
        *,
        heading: Float[Array, "T M"],
    ) -> "NumPyHeadings":
        """Creates a NumPy headings instance from an array of headings."""
        return NumPyHeadings(heading)

    def __array__(self) -> Float[Array, "T M"]:
        return self.array

    def heading(self) -> Float[Array, "T M"]:
        return self._heading

    @property
    def horizon(self) -> int:
        return self._heading.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._heading.shape[1]

    @property
    def array(self) -> Float[Array, "T M"]:
        return self._heading


@jaxtyped
@dataclass(frozen=True)
class NumPyReferencePoints(ReferencePoints):
    array: Float[Array, f"T {D_R} M"]

    @staticmethod
    def create(
        *, x: Float[Array, "T M"], y: Float[Array, "T M"], heading: Float[Array, "T M"]
    ) -> "NumPyReferencePoints":
        """Creates a NumPy reference points instance from x, y, and heading arrays."""
        return NumPyReferencePoints(array=np.stack([x, y, heading], axis=1))

    def __array__(self) -> Float[Array, f"T {D_R} M"]:
        return self.array

    def x(self) -> Float[Array, "T M"]:
        return self.array[:, 0]

    def y(self) -> Float[Array, "T M"]:
        return self.array[:, 1]

    def heading(self) -> Float[Array, "T M"]:
        return self.array[:, 2]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def positions(self) -> Float[Array, "T 2 M"]:
        return self.array[:, :2]


@jaxtyped
@dataclass(frozen=True)
class NumPyLateralPositions(LateralPositions):
    _array: Float[Array, "T M"]

    @staticmethod
    def create(array: Float[Array, "T M"]) -> "NumPyLateralPositions":
        """Creates a NumPy lateral positions instance from an array."""
        return NumPyLateralPositions(array)

    def __array__(self) -> Float[Array, "T M"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "T M"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyLongitudinalPositions(LongitudinalPositions):
    _array: Float[Array, "T M"]

    @staticmethod
    def create(array: Float[Array, "T M"]) -> "NumPyLongitudinalPositions":
        """Creates a NumPy longitudinal positions instance from an array."""
        return NumPyLongitudinalPositions(array)

    def __array__(self) -> Float[Array, "T M"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "T M"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyNormals(Normals):
    _array: Float[Array, "T 2 M"]

    @staticmethod
    def create(*, x: Float[Array, "T M"], y: Float[Array, "T M"]) -> "NumPyNormals":
        """Creates a NumPy normals instance from x and y coordinate arrays."""
        return NumPyNormals(np.stack([x, y], axis=1))

    def __array__(self) -> Float[Array, "T 2 M"]:
        return self.array

    def x(self) -> Float[Array, "T M"]:
        return self.array[:, 0]

    def y(self) -> Float[Array, "T M"]:
        return self.array[:, 1]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T 2 M"]:
        return self._array
