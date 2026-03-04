from typing import Protocol, overload
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

from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class JaxPathParameters(PathParameters):
    array: Float[JaxArray, "T M"]

    @overload
    @staticmethod
    def create(array: Float[Array, "T M"]) -> "JaxPathParameters":
        """Creates a JAX path parameters instance from a NumPy array."""
        ...

    @overload
    @staticmethod
    def create(array: Float[JaxArray, "T M"]) -> "JaxPathParameters":
        """Creates a JAX path parameters instance from a JAX array."""
        ...

    @staticmethod
    def create(
        array: Float[Array, "T M"] | Float[JaxArray, "T M"],
    ) -> "JaxPathParameters":
        return JaxPathParameters(array=jnp.asarray(array))

    def __array__(self) -> Float[Array, "T M"]:
        return np.asarray(self.array)

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]


class JaxPositions(Positions, Protocol):
    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        """Returns the x coordinates as a JAX array."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
        """Returns the y coordinates as a JAX array."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T 2 M"]:
        """Returns the positions as a JAX array."""
        ...


@jaxtyped
@dataclass(frozen=True)
class JaxSimplePositions(JaxPositions):
    _x_array: Float[JaxArray, "T M"]
    _y_array: Float[JaxArray, "T M"]

    @staticmethod
    def create(
        *,
        x: Float[JaxArray, "T M"] | Float[Array, "T M"],
        y: Float[JaxArray, "T M"] | Float[Array, "T M"],
    ) -> "JaxSimplePositions":
        """Creates a JAX positions instance from x and y coordinate arrays."""
        return JaxSimplePositions(_x_array=jnp.asarray(x), _y_array=jnp.asarray(y))

    def __array__(self) -> Float[Array, "T 2 M"]:
        return self._numpy_array

    def x(self) -> Float[Array, "T M"]:
        return np.asarray(self._x_array)

    def y(self) -> Float[Array, "T M"]:
        return np.asarray(self._y_array)

    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        return self._x_array

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
        return self._y_array

    @property
    def horizon(self) -> int:
        return self._x_array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._x_array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T 2 M"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "T 2 M"]:
        return jnp.stack([self._x_array, self._y_array], axis=1)

    @cached_property
    def _numpy_array(self) -> Float[Array, "T 2 M"]:
        return np.asarray(self._array)


@jaxtyped
@dataclass(frozen=True)
class JaxHeadings:
    _heading: Float[JaxArray, "T M"]

    @staticmethod
    def create(
        *, heading: Float[Array, "T M"] | Float[JaxArray, "T M"]
    ) -> "JaxHeadings":
        """Creates a JAX headings instance from an array of headings."""
        return JaxHeadings(jnp.asarray(heading))

    def __array__(self) -> Float[Array, "T M"]:
        return self._numpy_heading

    def heading(self) -> Float[Array, "T M"]:
        return self._numpy_heading

    @property
    def horizon(self) -> int:
        return self._heading.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._heading.shape[1]

    @property
    def heading_array(self) -> Float[JaxArray, "T M"]:
        return self._heading

    @cached_property
    def _numpy_heading(self) -> Float[Array, "T M"]:
        return np.asarray(self._heading)


@dataclass(frozen=True)
class JaxReferencePoints(ReferencePoints):
    array: Float[JaxArray, f"T {D_R} M"]

    @overload
    @staticmethod
    def create(
        *,
        x: Float[Array, "T M"],
        y: Float[Array, "T M"],
        heading: Float[Array, "T M"],
    ) -> "JaxReferencePoints":
        """Creates a JAX reference points instance from NumPy arrays."""
        ...

    @overload
    @staticmethod
    def create(
        *,
        x: Float[JaxArray, "T M"],
        y: Float[JaxArray, "T M"],
        heading: Float[JaxArray, "T M"],
    ) -> "JaxReferencePoints":
        """Creates a JAX reference points instance from JAX arrays."""
        ...

    @staticmethod
    def create(
        *,
        x: Float[Array, "T M"] | Float[JaxArray, "T M"],
        y: Float[Array, "T M"] | Float[JaxArray, "T M"],
        heading: Float[Array, "T M"] | Float[JaxArray, "T M"],
    ) -> "JaxReferencePoints":
        return JaxReferencePoints(array=jnp.stack([x, y, heading], axis=1))

    def __array__(self) -> Float[Array, f"T {D_R} M"]:
        return np.asarray(self.array)

    def x(self) -> Float[Array, "T M"]:
        return np.asarray(self.array[:, 0])

    def y(self) -> Float[Array, "T M"]:
        return np.asarray(self.array[:, 1])

    def heading(self) -> Float[Array, "T M"]:
        return np.asarray(self.array[:, 2])

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 0]

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 1]

    @property
    def heading_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 2]

    @property
    def positions_array(self) -> Float[JaxArray, "T 2 M"]:
        return self.array[:, :2]


@jaxtyped
@dataclass(frozen=True)
class JaxLateralPositions(LateralPositions):
    _array: Float[JaxArray, "T M"]

    @staticmethod
    def create(
        array: Float[JaxArray, "T M"] | Float[Array, "T M"],
    ) -> "JaxLateralPositions":
        """Creates a JAX lateral positions instance from an array."""
        return JaxLateralPositions(jnp.asarray(array))

    def __array__(self) -> Float[Array, "T M"]:
        return self._numpy_array

    @property
    def horizon(self) -> int:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T M"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxLongitudinalPositions(LongitudinalPositions):
    _array: Float[JaxArray, "T M"]

    @staticmethod
    def create(
        array: Float[JaxArray, "T M"] | Float[Array, "T M"],
    ) -> "JaxLongitudinalPositions":
        """Creates a JAX longitudinal positions instance from an array."""
        return JaxLongitudinalPositions(jnp.asarray(array))

    def __array__(self) -> Float[Array, "T M"]:
        return self._numpy_array

    @property
    def horizon(self) -> int:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T M"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxNormals(Normals):
    _array: Float[JaxArray, "T 2 M"]

    @staticmethod
    def create(
        *,
        x: Float[JaxArray, "T M"] | Float[Array, "T M"],
        y: Float[JaxArray, "T M"] | Float[Array, "T M"],
    ) -> "JaxNormals":
        """Creates a JAX normals instance from x and y coordinate arrays."""
        return JaxNormals(_array=jnp.stack([x, y], axis=1))

    def __array__(self) -> Float[Array, "T 2 M"]:
        return self._numpy_array

    def x(self) -> Float[Array, "T M"]:
        return np.asarray(self._array[:, 0])

    def y(self) -> Float[Array, "T M"]:
        return np.asarray(self._array[:, 1])

    @property
    def horizon(self) -> int:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self._array.shape[2]

    @property
    def array(self) -> Float[JaxArray, "T 2 M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T 2 M"]:
        return np.asarray(self.array)
