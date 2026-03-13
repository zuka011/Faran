from typing import Sequence, NamedTuple

from faran import (
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
    JaxGaussianBelief,
    JaxNoiseCovariances,
    NoiseModel,
    NoiseModelProvider,
    noise,
)

from numtypes import array, Array

import numpy as np
import jax.numpy as jnp

from tests.dsl import check, ArrayConvertible
from pytest import mark


class NoiseModelInputs[BeliefT, ObservationT](NamedTuple):
    observation: ObservationT
    prediction: BeliefT


def as_mean_state(observation: Array, *, state_dimension: int) -> Array:
    K = observation.shape[1]
    return np.vstack(
        [observation, np.zeros((state_dimension - observation.shape[0], K))]
    )


def some_mean_state_from(
    observation: Array,
    *,
    state_dimension: int,
    rng: np.random.Generator = np.random.default_rng(0),
) -> Array:
    return as_mean_state(
        observation + rng.normal(scale=2.0, size=observation.shape),
        state_dimension=state_dimension,
    )


def some_state_covariance(*, state_dimension: int, obstacle_count: int) -> Array:
    return np.tile(
        np.eye(state_dimension)[:, :, np.newaxis] * 0.01, (1, 1, obstacle_count)
    )


class test_that_noise_is_not_adapted_when_there_are_not_enough_observations:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        window = 5
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )

        def random_observations(count: int):
            return rng.normal(size=(count, D_z, 1))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    obstacle_count=(K := 1), observation_matrix=to_array(H), noise=noise
                ),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                as_mean_state(observation, state_dimension=D_x)
                            ),
                            covariance=to_array(
                                some_state_covariance(
                                    state_dimension=D_x, obstacle_count=K
                                )
                            ),
                        ),
                    )
                    for observation in random_observations(count)
                ],
            )
            for count in [1, window - 1]
        ]

    @mark.parametrize(
        ["noise", "model", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
    ) -> None:
        state = model.state
        for observation, prediction in inputs:
            result, state = model(
                noise=noise, prediction=prediction, observation=observation, state=state
            )

        assert np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_noise_is_adapted_when_there_are_enough_observations:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        observations = rng.normal(size=(window := 5, D_z, K := 1))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    obstacle_count=K, observation_matrix=to_array(H), noise=noise
                ),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                some_mean_state_from(
                                    observation, state_dimension=D_x, rng=rng
                                )
                            ),
                            covariance=to_array(
                                some_state_covariance(
                                    state_dimension=D_x, obstacle_count=K
                                )
                            ),
                        ),
                    )
                    for observation in observations
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
    ) -> None:
        state = model.state
        for observation, prediction in inputs:
            result, state = model(
                noise=noise, prediction=prediction, observation=observation, state=state
            )

        assert not np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert not np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_zero_innovation_produces_near_zero_adapted_noise:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, K := 1)
        )

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=1)(
                    obstacle_count=K, observation_matrix=to_array(H), noise=noise
                ),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                observation := to_array(H @ predicted_mean),
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "prediction", "observation"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        prediction: BeliefT,
        observation: ObservationT,
    ) -> None:
        result, _ = model(
            noise=noise,
            prediction=prediction,
            observation=observation,
            state=model.state,
        )

        R = np.asarray(result.process_noise_covariance)
        Q = np.asarray(result.observation_noise_covariance)

        # Anything less than 1e-7 is effectively zero for our purposes.
        assert np.allclose(R, 0.0, atol=1e-7), (
            "Adapted process noise covariance is not near zero for zero innovation."
        )
        assert np.allclose(Q, 0.0, atol=1e-7), (
            "Adapted observation noise covariance is not near zero for zero innovation."
        )

        # Still, it must be positive definite to avoid breaking the filter.
        assert check.is_spd(R, atol=1e-10), (
            "Adapted process noise covariance is not positive definite for zero innovation."
        )
        assert check.is_spd(Q, atol=1e-10), (
            "Adapted observation noise covariance is not positive definite for zero innovation."
        )


class test_that_adapted_process_noise_scales_quadratically_with_innovation:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, K := 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, K))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-10),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-10),
                ),
                observation_matrix := to_array(H),
                provider := provider(window_size=1),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                obstacle_count := K,
                observation_1 := to_array(H @ predicted_mean + innovation),
                observation_2 := to_array(H @ predicted_mean + 2 * innovation),
            ),
        ]

    @mark.parametrize(
        [
            "noise",
            "observation_matrix",
            "provider",
            "prediction",
            "obstacle_count",
            "observation_1",
            "observation_2",
        ],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        prediction: BeliefT,
        obstacle_count: int,
        observation_1: ObservationT,
        observation_2: ObservationT,
    ) -> None:
        model_1 = provider(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        model_2 = provider(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )

        result_1, _ = model_1(
            noise=noise,
            prediction=prediction,
            observation=observation_1,
            state=model_1.state,
        )
        result_2, _ = model_2(
            noise=noise,
            prediction=prediction,
            observation=observation_2,
            state=model_2.state,
        )

        assert np.allclose(
            np.asarray(result_2.process_noise_covariance),
            4.0 * np.asarray(result_1.process_noise_covariance),
            atol=1e-6,
        )


class test_that_all_innovation_is_attributed_to_observation_noise_when_state_is_certain:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, K := 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, K))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=1)(
                    obstacle_count=K, observation_matrix=to_array(H), noise=noise
                ),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 1e-12),
                ),
                observation := to_array(H @ predicted_mean + innovation),
                expected_observation_noise := to_array(innovation @ innovation.T),
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "prediction", "observation", "expected_observation_noise"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        prediction: BeliefT,
        observation: ObservationT,
        expected_observation_noise: ObservationT,
    ) -> None:
        result, _ = model(
            noise=noise,
            prediction=prediction,
            observation=observation,
            state=model.state,
        )

        assert np.allclose(np.asarray(result.process_noise_covariance), 0.0, atol=1e-6)
        assert np.allclose(
            np.asarray(result.observation_noise_covariance),
            np.asarray(expected_observation_noise),
            atol=1e-6,
        )


class test_that_repeated_identical_innovations_match_single_innovation:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, K := 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, K))
        observation = H @ predicted_mean + innovation

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x)),
                    observation_noise_covariance=to_array(np.eye(D_z)),
                ),
                observation_matrix := to_array(H),
                provider := provider,
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                obstacle_count := K,
                observation := to_array(observation),
            ),
        ]

    @mark.parametrize(
        [
            "noise",
            "observation_matrix",
            "provider",
            "prediction",
            "obstacle_count",
            "observation",
        ],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        prediction: BeliefT,
        obstacle_count: int,
        observation: ObservationT,
    ) -> None:
        single = provider(window_size=1)(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        result_single, _ = single(
            noise=noise,
            prediction=prediction,
            observation=observation,
            state=single.state,
        )

        repeated = provider(window_size=5)(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        state = repeated.state
        for _ in range(5):
            result_repeated, state = repeated(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
            )

        assert np.allclose(
            np.asarray(result_repeated.process_noise_covariance),
            np.asarray(result_single.process_noise_covariance),
        )
        assert np.allclose(
            np.asarray(result_repeated.observation_noise_covariance),
            np.asarray(result_single.observation_noise_covariance),
        )


class test_that_orthogonal_unit_innovations_produce_isotropic_adapted_process_noise:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], shape=(D_x, K := 1)
        )

        def unit_observation(*, dimensions: int, index: int):
            return to_array(np.eye(dimensions)[:, index : index + 1])

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-12),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-12),
                ),
                model := provider(window_size=D_z)(
                    obstacle_count=K, observation_matrix=to_array(H), noise=noise
                ),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                observations := [
                    unit_observation(dimensions=D_z, index=i) for i in range(D_z)
                ],
                observed_dimensions := D_z,
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "prediction", "observations", "observed_dimensions"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        prediction: BeliefT,
        observations: Sequence[ObservationT],
        observed_dimensions: int,
    ) -> None:
        state = model.state
        for observation in observations:
            result, state = model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
            )

        R = np.asarray(result.process_noise_covariance)
        observed_diagonal = np.diag(R)[:observed_dimensions]

        # Isotropic innovations → equal adapted noise for all observed states.
        assert np.allclose(observed_diagonal[0], observed_diagonal[1])
        assert np.allclose(observed_diagonal[0], observed_diagonal[2])

        # Observed states should have picked up some process noise.
        assert observed_diagonal[0] > 0

        # Unobserved states should have zero adapted process noise
        # (no information flows to them from isotropic observed innovations).
        unobserved_diagonal = np.diag(R)[observed_dimensions:]
        assert np.allclose(unobserved_diagonal, 0.0, atol=1e-7)


class test_that_two_models_from_same_provider_have_independent_state:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        window = 5
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        observations = rng.normal(size=(window + 1, D_z, K := 1))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                observation_matrix := to_array(H),
                provider := provider(window_size=window),
                obstacle_count := K,
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                as_mean_state(observation, state_dimension=D_x)
                            ),
                            covariance=to_array(
                                some_state_covariance(
                                    state_dimension=D_x, obstacle_count=K
                                )
                            ),
                        ),
                    )
                    for observation in observations
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "observation_matrix", "provider", "obstacle_count", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        obstacle_count: int,
        inputs: Sequence[NoiseModelInputs[ObservationT, BeliefT]],
    ) -> None:
        model_a = provider(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        model_b = provider(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )

        state_a = model_a.state
        for observation, prediction in inputs[:-1]:
            _, state_a = model_a(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_a,
            )

        # This one received less than `window` observations.
        observation, prediction = inputs[-1]
        state_b = model_b.state
        result, _ = model_b(
            noise=noise, prediction=prediction, observation=observation, state=state_b
        )

        # So we expect the noise to be unchanged.
        assert np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_adapted_noise_is_spd:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        observations = rng.normal(size=(window := 5, D_z, K := 1))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    obstacle_count=K, observation_matrix=to_array(H), noise=noise
                ),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                some_mean_state_from(
                                    observation, state_dimension=D_x, rng=rng
                                )
                            ),
                            covariance=to_array(
                                some_state_covariance(
                                    state_dimension=D_x, obstacle_count=K
                                )
                            ),
                        ),
                    )
                    for observation in observations
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs: Sequence[NoiseModelInputs[ObservationT, BeliefT]],
    ) -> None:
        state = model.state
        for observation, prediction in inputs:
            result, state = model(
                noise=noise, prediction=prediction, observation=observation, state=state
            )

        R = np.asarray(result.process_noise_covariance)
        Q = np.asarray(result.observation_noise_covariance)

        assert check.is_spd(R, atol=1e-7), (
            "Adapted process noise covariance is not positive definite"
        )
        assert check.is_spd(Q, atol=1e-7), (
            "Adapted observation noise covariance is not positive definite"
        )


class test_that_nan_prediction_returns_noise_unchanged:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=(window := 5))(
                    obstacle_count=(K := 1), observation_matrix=to_array(H), noise=noise
                ),
                nan_prediction := belief(
                    mean=to_array(np.full((D_x, K), np.nan)),
                    covariance=to_array(np.full((D_x, D_x, K), np.nan)),
                ),
                observation := to_array(np.zeros((D_z, K))),
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "nan_prediction", "observation"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        nan_prediction: BeliefT,
        observation: ObservationT,
    ) -> None:
        result, _ = model(
            noise=noise,
            prediction=nan_prediction,
            observation=observation,
            state=model.state,
        )

        assert np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_observations_older_than_window_size_do_not_affect_noise:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(42)
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )

        shared_observations = rng.normal(size=(window := 3, D_z, K := 1))
        old_observation_a = rng.normal(size=(D_z, K)) * 100.0
        old_observation_b = rng.normal(size=(D_z, K)) * 0.001

        def input_from(observation):
            return NoiseModelInputs(
                observation=to_array(observation),
                prediction=belief(
                    mean=to_array(np.zeros((D_x, K))),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 0.01),
                ),
            )

        return [
            (
                initial_noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x)),
                    observation_noise_covariance=to_array(np.eye(D_z)),
                ),
                model := provider(window_size=window)(
                    obstacle_count=K,
                    observation_matrix=to_array(H),
                    noise=initial_noise,
                ),
                inputs_a := [
                    input_from(old_observation_a),
                    *[input_from(it) for it in shared_observations],
                ],
                inputs_b := [
                    input_from(old_observation_b),
                    *[input_from(it) for it in shared_observations],
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "inputs_a", "inputs_b"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs_a: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
        inputs_b: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
    ) -> None:
        state_a = model.state
        for observation, prediction in inputs_a:
            result_a, state_a = model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_a,
            )

        state_b = model.state
        for observation, prediction in inputs_b:
            result_b, state_b = model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_b,
            )

        assert np.allclose(
            result_a.process_noise_covariance,
            result_b.process_noise_covariance,
            atol=1e-6,
        )
        assert np.allclose(
            result_a.observation_noise_covariance,
            result_b.observation_noise_covariance,
            atol=1e-6,
        )


class test_that_noise_is_adapted_when_not_all_obstacles_have_enough_observations:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        history = rng.normal(size=(window := 5, D_z, 1))

        def some_mean_estimate(*, state_dimension: int, obstacle_count: int):
            return np.random.default_rng(1).normal(
                size=(state_dimension, obstacle_count)
            )

        def missing_estimate():
            return np.full((D_x, 1), np.nan)

        def missing_covariance():
            return np.full((D_x, D_x, 1), np.nan)

        def single_obstacle_inputs(observations, *, rng):
            return [
                NoiseModelInputs(
                    observation=to_array(observation),
                    prediction=belief(
                        mean=to_array(
                            some_mean_state_from(
                                observation, state_dimension=D_x, rng=rng
                            )
                        ),
                        covariance=to_array(
                            some_state_covariance(state_dimension=D_x, obstacle_count=1)
                        ),
                    ),
                )
                for observation in observations
            ]

        def multiple_obstacle_inputs(
            observations,
            *,
            rng,
            with_missing_observation: bool = False,
            with_missing_estimate: bool = False,
            with_missing_covariance: bool = False,
        ):
            return [
                NoiseModelInputs(
                    observation=to_array(
                        np.hstack(
                            [
                                observation,
                                np.full((D_z, 1), np.nan)
                                if with_missing_observation
                                else np.zeros((D_z, 1)),
                            ]
                        )
                    ),
                    prediction=belief(
                        mean=to_array(
                            np.hstack(
                                [
                                    some_mean_state_from(
                                        observation, state_dimension=D_x, rng=rng
                                    ),
                                    missing_estimate()
                                    if with_missing_estimate
                                    else some_mean_estimate(
                                        state_dimension=D_x, obstacle_count=1
                                    ),
                                ]
                            )
                        ),
                        covariance=to_array(
                            np.concatenate(
                                [
                                    some_state_covariance(
                                        state_dimension=D_x, obstacle_count=1
                                    ),
                                    missing_covariance()
                                    if with_missing_covariance
                                    else some_state_covariance(
                                        state_dimension=D_x, obstacle_count=1
                                    ),
                                ],
                                axis=2,
                            )
                        ),
                    ),
                )
                for observation in observations
            ]

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                observation_matrix := to_array(H),
                provider(window_size=window),
                inputs := single_obstacle_inputs(history, rng=np.random.default_rng(1)),
                inputs_with_missing_data,
            )
            for inputs_with_missing_data in [
                multiple_obstacle_inputs(
                    history, rng=np.random.default_rng(1), with_missing_estimate=True
                ),
                multiple_obstacle_inputs(
                    history, rng=np.random.default_rng(1), with_missing_covariance=True
                ),
                multiple_obstacle_inputs(
                    history, rng=np.random.default_rng(1), with_missing_observation=True
                ),
            ]
        ]

    @mark.parametrize(
        [
            "noise",
            "observation_matrix",
            "provider",
            "inputs",
            "inputs_with_missing_data",
        ],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        inputs: Sequence[NoiseModelInputs[ObservationT, BeliefT]],
        inputs_with_missing_data: Sequence[NoiseModelInputs[ObservationT, BeliefT]],
    ) -> None:
        model_1 = provider(
            obstacle_count=1, observation_matrix=observation_matrix, noise=noise
        )
        state_1 = model_1.state
        for observation, prediction in inputs:
            result_1, state_1 = model_1(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_1,
            )

        model_2 = provider(
            obstacle_count=2, observation_matrix=observation_matrix, noise=noise
        )
        state_2 = model_2.state
        for observation, prediction in inputs_with_missing_data:
            result_2, state_2 = model_2(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_2,
            )

        assert np.allclose(
            result_1.process_noise_covariance[..., :1],
            result_2.process_noise_covariance[..., :1],
            atol=1e-6,
        ), (
            f"Expected adapted noise covariance for present obstacle to be: {result_1.process_noise_covariance[..., :1]}, "
            f"but got: {result_2.process_noise_covariance[..., :1]}"
        )

        assert np.allclose(
            result_1.observation_noise_covariance,
            result_2.observation_noise_covariance,
            atol=1e-6,
        ), (
            f"Expected adapted observation noise covariance to be: {result_1.observation_noise_covariance}, "
            f"but got: {result_2.observation_noise_covariance}"
        )


class test_that_adapted_noise_is_robust_to_single_outlier_innovation:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, K := 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, K))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-10),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-10),
                ),
                observation_matrix := to_array(H),
                provider := provider(window_size=5),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                obstacle_count := K,
                normal_observation := to_array(H @ predicted_mean + innovation),
                outlier_observation := to_array(
                    H @ predicted_mean + innovation * 1000.0
                ),
            ),
        ]

    @mark.parametrize(
        [
            "noise",
            "observation_matrix",
            "provider",
            "prediction",
            "obstacle_count",
            "normal_observation",
            "outlier_observation",
        ],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        prediction: BeliefT,
        obstacle_count: int,
        normal_observation: ObservationT,
        outlier_observation: ObservationT,
    ) -> None:
        clean_model = provider(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        state = clean_model.state
        for observation in [normal_observation] * 5:
            result_clean, state = clean_model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
            )

        outlier_model = provider(
            obstacle_count=obstacle_count,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        state = outlier_model.state
        for observation in [normal_observation] * 4 + [outlier_observation]:
            result_outlier, state = outlier_model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
            )

        assert np.allclose(
            np.asarray(result_outlier.process_noise_covariance),
            np.asarray(result_clean.process_noise_covariance),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(result_outlier.observation_noise_covariance),
            np.asarray(result_clean.observation_noise_covariance),
            atol=1e-6,
        )


class test_that_adapted_noise_covariance_for_multiple_obstacles_is_average_of_individual_adapted_covariances:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, K := 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, K))
        observation = H @ predicted_mean + innovation

        covariance_confident = np.eye(D_x)[:, :, np.newaxis] * 1e-6
        covariance_uncertain = np.eye(D_x)[:, :, np.newaxis] * 100.0

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-10),
                    observation_noise_covariance=to_array(np.eye(D_z)),
                ),
                observation_matrix := to_array(H),
                provider := provider(window_size=1),
                prediction_a := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(covariance_confident),
                ),
                prediction_b := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(covariance_uncertain),
                ),
                prediction_combined := belief(
                    mean=to_array(np.hstack([predicted_mean, predicted_mean])),
                    covariance=to_array(
                        np.concatenate(
                            [covariance_confident, covariance_uncertain], axis=2
                        )
                    ),
                ),
                observation_single := to_array(observation),
                observation_combined := to_array(np.hstack([observation, observation])),
            ),
        ]

    @mark.parametrize(
        [
            "noise",
            "observation_matrix",
            "provider",
            "prediction_a",
            "prediction_b",
            "prediction_combined",
            "observation_single",
            "observation_combined",
        ],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        prediction_a: BeliefT,
        prediction_b: BeliefT,
        prediction_combined: BeliefT,
        observation_single: ObservationT,
        observation_combined: ObservationT,
    ) -> None:
        model_a = provider(
            obstacle_count=1,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        result_a, _ = model_a(
            noise=noise,
            prediction=prediction_a,
            observation=observation_single,
            state=model_a.state,
        )

        model_b = provider(
            obstacle_count=1,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        result_b, _ = model_b(
            noise=noise,
            prediction=prediction_b,
            observation=observation_single,
            state=model_b.state,
        )

        model_combined = provider(
            obstacle_count=2,
            observation_matrix=observation_matrix,
            noise=noise,
        )
        result_combined, _ = model_combined(
            noise=noise,
            prediction=prediction_combined,
            observation=observation_combined,
            state=model_combined.state,
        )

        expected_process = (
            np.asarray(result_a.process_noise_covariance)
            + np.asarray(result_b.process_noise_covariance)
        ) / 2
        expected_observation = (
            np.asarray(result_a.observation_noise_covariance)
            + np.asarray(result_b.observation_noise_covariance)
        ) / 2

        assert np.allclose(
            np.asarray(result_combined.process_noise_covariance),
            expected_process,
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(result_combined.observation_noise_covariance),
            expected_observation,
            atol=1e-6,
        )


class test_that_noise_covariances_are_created_correctly:
    @staticmethod
    def cases(covariances, to_array) -> Sequence[tuple]:
        return [
            (  # Entire covariance matrices specified directly
                covariances(
                    process=to_array(process := np.array([[1.0, 0.5], [0.5, 2.0]])),
                    observation=to_array(
                        observation := np.array(
                            [[3.0, 0.1, 0.0], [0.1, 4.0, 0.2], [0.0, 0.2, 5.0]]
                        )
                    ),
                    process_dimension=2,
                    observation_dimension=3,
                ),
                expected_process := process,
                expected_observation := observation,
            ),
            (  # Diagonal vectors specified
                covariances(
                    process=to_array(np.array([1.0, 2.0, 3.0])),
                    observation=to_array(np.array([4.0, 5.0])),
                    process_dimension=3,
                    observation_dimension=2,
                ),
                expected_process := np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0],
                    ]
                ),
                expected_observation := np.array(
                    [
                        [4.0, 0.0],
                        [0.0, 5.0],
                    ]
                ),
            ),
            (  # Scalar variance values and dimensions
                covariances(
                    process=0.01,
                    observation=0.5,
                    process_dimension=4,
                    observation_dimension=2,
                ),
                expected_process := np.array(
                    [
                        [0.01, 0.0, 0.0, 0.0],
                        [0.0, 0.01, 0.0, 0.0],
                        [0.0, 0.0, 0.01, 0.0],
                        [0.0, 0.0, 0.0, 0.01],
                    ]
                ),
                expected_observation := np.array(
                    [
                        [0.5, 0.0],
                        [0.0, 0.5],
                    ]
                ),
            ),
        ]

    @mark.parametrize(
        ["result", "expected_process", "expected_observation"],
        [
            *cases(covariances=noise.numpy.covariances, to_array=np.asarray),
            *cases(covariances=noise.jax.covariances, to_array=jnp.asarray),
        ],
    )
    def test(
        self,
        result: NumPyNoiseCovariances | JaxNoiseCovariances,
        expected_process: ArrayConvertible,
        expected_observation: ArrayConvertible,
    ) -> None:
        assert np.allclose(result.process_noise_covariance, expected_process)
        assert np.allclose(result.observation_noise_covariance, expected_observation)
