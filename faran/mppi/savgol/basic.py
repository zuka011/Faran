from dataclasses import dataclass

from faran.types import Array, jaxtyped
from faran.types import NumPyControlInputSequence

from jaxtyping import Float
from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d


@jaxtyped
@dataclass(frozen=True)
class NumPySavGolFilter:
    """Savitzky-Golay smoothing filter for MPPI control sequences (NumPy)."""

    coefficients: Float[Array, " W"]

    @staticmethod
    def create(*, window_length: int, polynomial_order: int) -> "NumPySavGolFilter":
        assert window_length % 2 == 1, f"Window length must be odd, got {window_length}"
        assert 0 <= polynomial_order < window_length, (
            f"Polynomial order must be non-negative and less than window length, got {polynomial_order}"
        )

        coefficients = savgol_coeffs(window_length, polynomial_order)

        assert coefficients.shape == (window_length,)

        return NumPySavGolFilter(coefficients=coefficients)

    def __call__[InputSequenceT: NumPyControlInputSequence](
        self, *, optimal_input: InputSequenceT
    ) -> InputSequenceT:
        filtered = convolve1d(
            optimal_input.array, weights=self.coefficients, axis=0, mode="nearest"
        )

        assert filtered.shape == optimal_input.array.shape

        return optimal_input.similar(array=filtered)
