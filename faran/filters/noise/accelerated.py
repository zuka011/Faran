from typing import NamedTuple

from faran.types import (
    jaxtyped,
    JaxNoiseModel,
    JaxNoiseModelProvider,
    JaxGaussianBelief,
    JaxNoiseCovariances,
)

from jaxtyping import Array as JaxArray, Bool, Float, Int, Scalar

import equinox as eqx
import jax
import jax.numpy as jnp


class JaxNoiseCovarianceBounds(NamedTuple):
    process: Scalar
    observation: Scalar


class JaxClampedNoise[StateT](eqx.Module):
    """Decorator that clamps an inner noise model's eigenvalues to a floor and/or ceiling."""

    inner: JaxNoiseModel[StateT]
    floor: JaxNoiseCovarianceBounds
    ceiling: JaxNoiseCovarianceBounds

    @eqx.filter_jit
    @jaxtyped
    def __call__(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: StateT,
    ) -> tuple[JaxNoiseCovariances, StateT]:
        result, state = self.inner(
            noise=noise, prediction=prediction, observation=observation, state=state
        )
        return JaxNoiseCovariances(
            process_noise_covariance=clamp_eigenvalues(
                result.process_noise_covariance,
                floor=self.floor.process,
                ceiling=self.ceiling.process,
            ),
            observation_noise_covariance=clamp_eigenvalues(
                result.observation_noise_covariance,
                floor=self.floor.observation,
                ceiling=self.ceiling.observation,
            ),
        ), state

    @property
    def state(self) -> StateT:
        return self.inner.state


class JaxClampedNoiseProvider[StateT](eqx.Module):
    inner: JaxNoiseModelProvider[StateT]
    floor: JaxNoiseCovarianceBounds
    ceiling: JaxNoiseCovarianceBounds

    @staticmethod
    def decorate[S](
        inner: JaxNoiseModelProvider[S],
        *,
        floor: JaxNoiseCovarianceBounds | None = None,
        ceiling: JaxNoiseCovarianceBounds | None = None,
    ) -> "JaxClampedNoiseProvider[S]":
        """Creates a noise model provider that clamps the eigenvalues of the
        noise covariances to the specified floor and/or ceiling.

        Args:
            inner: The inner noise model provider to delegate to.
            floor: Isotropic minimum bounds. Eigenvalues of the inner model's
                output will be clamped to be no smaller than these.
            ceiling: Isotropic maximum bounds. Eigenvalues of the inner model's
                output will be clamped to be no larger than these.
        """
        return JaxClampedNoiseProvider(
            inner=inner,
            floor=floor
            or JaxNoiseCovarianceBounds(
                process=jnp.asarray(0.0), observation=jnp.asarray(0.0)
            ),
            ceiling=ceiling
            or JaxNoiseCovarianceBounds(
                process=jnp.asarray(jnp.inf), observation=jnp.asarray(jnp.inf)
            ),
        )

    @eqx.filter_jit
    def __call__(
        self,
        *,
        obstacle_count: int,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        noise: JaxNoiseCovariances,
    ) -> JaxClampedNoise:
        inner_model = self.inner(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        return JaxClampedNoise(
            inner=inner_model, floor=self.floor, ceiling=self.ceiling
        )


class JaxAdaptiveNoiseState(NamedTuple):
    """Circular buffer state for adaptive noise estimation."""

    buffer: Float[JaxArray, "W D_z D_z K"]
    entry_count: Int[JaxArray, " K"]


class JaxAdaptiveNoise(eqx.Module):
    """Innovation-Based Adaptive Estimation (IAE) for noise covariances."""

    observation_matrix: Float[JaxArray, "D_z D_x"]
    obstacle_count: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)

    @eqx.filter_jit
    @jaxtyped
    def __call__(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: JaxAdaptiveNoiseState,
    ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
        valid = valid_obstacle_mask(prediction, observation)
        has_valid = jnp.any(valid)

        return jax.lax.cond(
            has_valid,
            lambda _: self.adapt(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
                valid=valid,
            ),
            lambda _: (noise, state),
            None,
        )

    @eqx.filter_jit
    @jaxtyped
    def adapt(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: JaxAdaptiveNoiseState,
        valid: Bool[JaxArray, " K"],
    ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
        innovation = compute_innovation_matrix(
            prediction=prediction,
            observation=observation,
            observation_matrix=self.observation_matrix,
        )
        new_state = compute_updated_state(
            state=state, innovation=innovation, valid=valid
        )

        any_buffer_full = jnp.any(
            buffer_full := new_state.entry_count >= self.window_size
        )

        def return_original(
            _: None,
        ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
            return noise, new_state

        def compute_adapted(
            _: None,
        ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
            innovation_matrices = jnp.median(new_state.buffer, axis=0)
            safe_covariance = jnp.where(
                jnp.isnan(prediction.covariance), 0.0, prediction.covariance
            )

            kalman_gains = jax.vmap(
                lambda cov: compute_kalman_gain(
                    predicted_covariance=cov,
                    observation_matrix=self.observation_matrix,
                    observation_noise_covariance=noise.observation_noise_covariance,
                ),
                in_axes=2,
                out_axes=2,
            )(safe_covariance)

            adapted_process = jax.vmap(
                lambda K, V: enforce_spd(K @ V @ K.T),
                in_axes=(2, 2),
                out_axes=2,
            )(kalman_gains, innovation_matrices)

            adapted_observation = jax.vmap(
                lambda V, P: enforce_spd(
                    V - self.observation_matrix @ P @ self.observation_matrix.T
                ),
                in_axes=(2, 2),
                out_axes=2,
            )(innovation_matrices, safe_covariance)

            adapted_noise = aggregate_noise_covariances(
                buffer_full=buffer_full,
                valid=valid,
                adapted_process=adapted_process,
                adapted_observation=adapted_observation,
                noise=noise,
            )

            return adapted_noise, new_state

        return jax.lax.cond(any_buffer_full, compute_adapted, return_original, None)

    @property
    def state(self) -> JaxAdaptiveNoiseState:
        observation_dimension = self.observation_matrix.shape[0]
        return JaxAdaptiveNoiseState(
            buffer=jnp.zeros(
                (
                    self.window_size,
                    observation_dimension,
                    observation_dimension,
                    self.obstacle_count,
                )
            ),
            entry_count=jnp.zeros(self.obstacle_count, dtype=jnp.int32),
        )


class JaxAdaptiveNoiseProvider(eqx.Module):
    window_size: int = eqx.field(static=True)

    @staticmethod
    def create(*, window_size: int) -> "JaxAdaptiveNoiseProvider":
        """Creates an innovation-based adaptive estimation model for noise.

        Args:
            window_size: The number of past observations considered, when adapting
                the noise covariances.
        """
        return JaxAdaptiveNoiseProvider(window_size=window_size)

    def __call__(
        self,
        *,
        obstacle_count: int,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        noise: JaxNoiseCovariances,
    ) -> JaxAdaptiveNoise:
        return JaxAdaptiveNoise(
            observation_matrix=observation_matrix,
            obstacle_count=obstacle_count,
            window_size=self.window_size,
        )


def valid_obstacle_mask(
    prediction: JaxGaussianBelief, observation: Float[JaxArray, "D_z K"]
) -> Bool[JaxArray, " K"]:
    mean_valid = ~jnp.any(jnp.isnan(prediction.mean), axis=0)
    covariance_valid = ~jnp.any(
        jnp.isnan(prediction.covariance.reshape(-1, prediction.covariance.shape[2])),
        axis=0,
    )
    observation_valid = ~jnp.any(jnp.isnan(observation), axis=0)
    return mean_valid & covariance_valid & observation_valid


@jax.jit
@jaxtyped
def compute_innovation_matrix(
    prediction: JaxGaussianBelief,
    observation: Float[JaxArray, "D_z K"],
    *,
    observation_matrix: Float[JaxArray, "D_z D_x"],
) -> Float[JaxArray, "D_z D_z K"]:
    safe_mean = jnp.where(jnp.isnan(prediction.mean), 0.0, prediction.mean)
    safe_observation = jnp.where(jnp.isnan(observation), 0.0, observation)
    innovation = safe_observation - observation_matrix @ safe_mean

    return innovation[:, jnp.newaxis, :] * innovation[jnp.newaxis, :, :]


@jax.jit
@jaxtyped
def compute_updated_state(
    state: JaxAdaptiveNoiseState,
    *,
    innovation: Float[JaxArray, "D_z D_z K"],
    valid: Bool[JaxArray, " K"],
) -> JaxAdaptiveNoiseState:
    window_size = state.buffer.shape[0]
    obstacle_count = valid.shape[0]

    indices = state.entry_count % window_size
    updated_buffer = state.buffer.at[indices, :, :, jnp.arange(obstacle_count)].set(
        innovation.transpose(2, 0, 1)
    )

    return JaxAdaptiveNoiseState(
        buffer=jnp.where(
            valid[jnp.newaxis, jnp.newaxis, jnp.newaxis, :],
            updated_buffer,
            state.buffer,
        ),
        entry_count=state.entry_count + valid.astype(jnp.int32),
    )


@jax.jit
@jaxtyped
def aggregate_noise_covariances(
    *,
    buffer_full: Bool[JaxArray, " K"],
    valid: Bool[JaxArray, " K"],
    adapted_process: Float[JaxArray, "D_x D_x K"],
    adapted_observation: Float[JaxArray, "D_z D_z K"],
    noise: JaxNoiseCovariances,
) -> JaxNoiseCovariances:
    use_adapted = buffer_full & valid
    valid_adapted_count = jnp.maximum(jnp.sum(use_adapted), 1)

    mean_process = (
        jnp.sum(
            jnp.where(
                use_adapted[jnp.newaxis, jnp.newaxis, :],
                adapted_process,
                0.0,
            ),
            axis=2,
        )
        / valid_adapted_count
    )

    mean_observation = (
        jnp.sum(
            jnp.where(
                use_adapted[jnp.newaxis, jnp.newaxis, :],
                adapted_observation,
                0.0,
            ),
            axis=2,
        )
        / valid_adapted_count
    )

    has_valid_adapted = jnp.any(use_adapted)

    return JaxNoiseCovariances(
        process_noise_covariance=jnp.where(
            has_valid_adapted,
            mean_process,
            noise.process_noise_covariance,
        ),
        observation_noise_covariance=jnp.where(
            has_valid_adapted,
            mean_observation,
            noise.observation_noise_covariance,
        ),
    )


@jax.jit
@jaxtyped
def compute_kalman_gain(
    *,
    predicted_covariance: Float[JaxArray, "D_x D_x"],
    observation_matrix: Float[JaxArray, "D_z D_x"],
    observation_noise_covariance: Float[JaxArray, "D_z D_z"],
) -> Float[JaxArray, "D_x D_z"]:
    S = (
        observation_matrix @ predicted_covariance @ observation_matrix.T
        + observation_noise_covariance
    )
    return jnp.linalg.solve(S, observation_matrix @ predicted_covariance).T


@jax.jit
@jaxtyped
def clamp_eigenvalues(
    matrix: Float[JaxArray, "N N"], *, floor: Scalar, ceiling: Scalar
) -> Float[JaxArray, "N N"]:
    symmetrised = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetrised)
    clamped = jnp.clip(eigenvalues, floor, ceiling)
    return (eigenvectors * clamped[jnp.newaxis, :]) @ eigenvectors.T


@jax.jit
@jaxtyped
def enforce_spd(matrix: Float[JaxArray, "N N"]) -> Float[JaxArray, "N N"]:
    eps = 1e-8
    symmetrised = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetrised)
    return eigenvectors @ jnp.diag(jnp.maximum(eigenvalues, eps)) @ eigenvectors.T
