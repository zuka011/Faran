from typing import NamedTuple

from faran.types import (
    Array,
    NumPyNoiseModel,
    NumPyNoiseModelProvider,
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
)

from jaxtyping import Bool, Float

import numpy as np


class NumPyClampedNoise[StateT](NamedTuple):
    """Decorator that clamps an inner noise model's output diagonals to a floor."""

    inner: NumPyNoiseModel
    floor: NumPyNoiseCovariances

    def __call__(
        self,
        *,
        noise: NumPyNoiseCovariances,
        prediction: NumPyGaussianBelief,
        observation: Float[Array, "D_z K"],
        state: StateT,
    ) -> tuple[NumPyNoiseCovariances, StateT]:
        result, state = self.inner(
            noise=noise, prediction=prediction, observation=observation, state=state
        )
        return NumPyNoiseCovariances(
            process_noise_covariance=apply_diagonal_floor(
                result.process_noise_covariance,
                floor=self.floor.process_noise_covariance,
            ),
            observation_noise_covariance=apply_diagonal_floor(
                result.observation_noise_covariance,
                floor=self.floor.observation_noise_covariance,
            ),
        ), state

    @property
    def state(self) -> StateT:
        return self.inner.state


class NumPyClampedNoiseProvider[StateT](NamedTuple):
    inner: NumPyNoiseModelProvider[StateT]
    floor: NumPyNoiseCovariances

    @staticmethod
    def decorate[S](
        inner: NumPyNoiseModelProvider[S], *, floor: NumPyNoiseCovariances
    ) -> "NumPyClampedNoiseProvider[S]":
        """Creates a noise model provider that clamps the diagonal of the
        noise covariances to the specified floor.

        Args:
            inner: The inner noise model provider to delegate to.
            floor: Minimum noise covariances. Diagonal entries of the inner model's
                output will be clamped to be no smaller than these.
        """
        return NumPyClampedNoiseProvider(floor=floor, inner=inner)

    def __call__(
        self,
        *,
        obstacle_count: int,
        observation_matrix: Float[Array, "D_z D_x"],
        noise: NumPyNoiseCovariances,
    ) -> NumPyClampedNoise:
        inner_model = self.inner(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        return NumPyClampedNoise(inner=inner_model, floor=self.floor)


class NumPyAdaptiveNoiseState(NamedTuple):
    """Circular buffer state for adaptive noise estimation."""

    buffer: Float[Array, "W D_z D_z K"]
    entry_count: Float[Array, " K"]


class NumPyAdaptiveNoise(NamedTuple):
    """Innovation-Based Adaptive Estimation (IAE) for noise covariances."""

    observation_matrix: Float[Array, "D_z D_x"]
    obstacle_count: int
    window_size: int

    def __call__(
        self,
        *,
        noise: NumPyNoiseCovariances,
        prediction: NumPyGaussianBelief,
        observation: Float[Array, "D_z K"],
        state: NumPyAdaptiveNoiseState,
    ) -> tuple[NumPyNoiseCovariances, NumPyAdaptiveNoiseState]:
        if not np.any(valid := valid_obstacle_mask(prediction, observation)):
            return noise, state

        innovation = compute_innovation(
            prediction=prediction,
            observation=observation,
            observation_matrix=self.observation_matrix,
        )
        new_state = compute_updated_state(
            state=state, innovation=innovation, valid=valid
        )

        if not np.any(buffer_full := new_state.entry_count >= self.window_size):
            return noise, new_state

        innovation_matrices = np.median(new_state.buffer, axis=0)
        safe_covariance = np.where(
            np.isnan(prediction.covariance), 0.0, prediction.covariance
        )

        P = safe_covariance.transpose(2, 0, 1)
        V = innovation_matrices.transpose(2, 0, 1)
        H = self.observation_matrix

        kalman_gains = compute_kalman_gain(
            predicted_covariance=P,
            observation_matrix=H,
            observation_noise_covariance=noise.observation_noise_covariance,
        )

        adapted_process = enforce_spd(
            kalman_gains @ V @ kalman_gains.swapaxes(-2, -1)
        ).transpose(1, 2, 0)

        adapted_observation = enforce_spd(V - H @ P @ H.T).transpose(1, 2, 0)

        adapted_noise = aggregate_noise_covariances(
            buffer_full=buffer_full,
            valid=valid,
            adapted_process=adapted_process,
            adapted_observation=adapted_observation,
            noise=noise,
        )

        return adapted_noise, new_state

    @property
    def state(self) -> NumPyAdaptiveNoiseState:
        observation_dimension = self.observation_matrix.shape[0]
        return NumPyAdaptiveNoiseState(
            buffer=np.zeros(
                (
                    self.window_size,
                    observation_dimension,
                    observation_dimension,
                    self.obstacle_count,
                )
            ),
            entry_count=np.zeros(self.obstacle_count, dtype=np.int32),
        )


class NumPyAdaptiveNoiseProvider(NamedTuple):
    window_size: int

    @staticmethod
    def create(*, window_size: int) -> "NumPyAdaptiveNoiseProvider":
        """Creates an innovation-based adaptive estimation model for noise.

        Args:
            window_size: The number of past observations considered, when adapting
                the noise covariances.
        """
        return NumPyAdaptiveNoiseProvider(window_size=window_size)

    def __call__(
        self,
        *,
        obstacle_count: int,
        observation_matrix: Float[Array, "D_z D_x"],
        noise: NumPyNoiseCovariances,
    ) -> NumPyAdaptiveNoise:
        return NumPyAdaptiveNoise(
            observation_matrix=observation_matrix,
            obstacle_count=obstacle_count,
            window_size=self.window_size,
        )


def valid_obstacle_mask(
    prediction: NumPyGaussianBelief, observation: Float[Array, "D_z K"]
) -> Bool[Array, " K"]:
    mean_valid = ~np.any(np.isnan(prediction.mean), axis=0)
    covariance_valid = ~np.any(
        np.isnan(prediction.covariance.reshape(-1, prediction.covariance.shape[2])),
        axis=0,
    )
    observation_valid = ~np.any(np.isnan(observation), axis=0)
    return mean_valid & covariance_valid & observation_valid


def compute_innovation(
    prediction: NumPyGaussianBelief,
    observation: Float[Array, "D_z K"],
    *,
    observation_matrix: Float[Array, "D_z D_x"],
) -> Float[Array, "D_z D_z K"]:
    safe_mean = np.where(np.isnan(prediction.mean), 0.0, prediction.mean)
    safe_observation = np.where(np.isnan(observation), 0.0, observation)
    innovation = safe_observation - observation_matrix @ safe_mean

    return innovation[:, np.newaxis, :] * innovation[np.newaxis, :, :]


def compute_updated_state(
    state: NumPyAdaptiveNoiseState,
    *,
    innovation: Float[Array, "D_z D_z K"],
    valid: Bool[Array, " K"],
) -> NumPyAdaptiveNoiseState:
    window_size = state.buffer.shape[0]
    obstacle_count = valid.shape[0]

    indices = state.entry_count % window_size
    updated_buffer = state.buffer.copy()
    updated_buffer[indices, :, :, np.arange(obstacle_count)] = innovation.transpose(
        2, 0, 1
    )

    return NumPyAdaptiveNoiseState(
        buffer=np.where(
            valid[np.newaxis, np.newaxis, np.newaxis, :],
            updated_buffer,
            state.buffer,
        ),
        entry_count=state.entry_count + valid.astype(np.int32),
    )


def aggregate_noise_covariances(
    *,
    buffer_full: Bool[Array, " K"],
    valid: Bool[Array, " K"],
    adapted_process: Float[Array, "D_x D_x K"],
    adapted_observation: Float[Array, "D_z D_z K"],
    noise: NumPyNoiseCovariances,
) -> NumPyNoiseCovariances:
    use_adapted = buffer_full & valid
    valid_adapted_count = max(np.sum(use_adapted), 1)

    mean_process = (
        np.sum(
            np.where(
                use_adapted[np.newaxis, np.newaxis, :],
                adapted_process,
                0.0,
            ),
            axis=2,
        )
        / valid_adapted_count
    )

    mean_observation = (
        np.sum(
            np.where(
                use_adapted[np.newaxis, np.newaxis, :],
                adapted_observation,
                0.0,
            ),
            axis=2,
        )
        / valid_adapted_count
    )

    has_valid_adapted = np.any(use_adapted)

    return NumPyNoiseCovariances(
        process_noise_covariance=np.where(
            has_valid_adapted,
            mean_process,
            noise.process_noise_covariance,
        ),
        observation_noise_covariance=np.where(
            has_valid_adapted,
            mean_observation,
            noise.observation_noise_covariance,
        ),
    )


def compute_kalman_gain(
    *,
    predicted_covariance: Float[Array, "... D_x D_x"],
    observation_matrix: Float[Array, "D_z D_x"],
    observation_noise_covariance: Float[Array, "D_z D_z"],
) -> Float[Array, "... D_x D_z"]:
    S = (
        observation_matrix @ predicted_covariance @ observation_matrix.T
        + observation_noise_covariance
    )
    return np.linalg.solve(S, observation_matrix @ predicted_covariance).swapaxes(
        -2, -1
    )


def enforce_spd(matrix: Float[Array, "... N N"]) -> Float[Array, "... N N"]:
    eps = 1e-8
    symmetrised = (matrix + matrix.swapaxes(-2, -1)) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrised)
    return (
        eigenvectors * np.maximum(eigenvalues, eps)[..., np.newaxis, :]
    ) @ eigenvectors.swapaxes(-2, -1)


def apply_diagonal_floor(
    matrix: Float[Array, "N N"], *, floor: Float[Array, "N N"]
) -> Float[Array, "N N"]:
    floored = np.maximum(np.diag(matrix), np.diag(floor))
    return matrix - np.diag(np.diag(matrix)) + np.diag(floored)
