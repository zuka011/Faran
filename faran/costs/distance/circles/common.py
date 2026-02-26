from dataclasses import dataclass

from faran.types.array import Array, jaxtyped

from jaxtyping import Float

type OriginsArray = Float[Array, "V 2"]
type RadiiArray = Float[Array, " V"]


@jaxtyped
@dataclass(frozen=True)
class Circles:
    """Describes circles approximating parts of an object for distance computation."""

    origins: OriginsArray
    """The local (x, y) offset of the circle center from the object center."""

    radii: RadiiArray
    """The radius of the circle."""
