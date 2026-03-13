from typing import Sequence

from faran import Noise, NoiseModel, NumPyGaussianBelief, JaxGaussianBelief, noise

import numpy as np
import jax.numpy as jnp

from tests.dsl import stubs, check, compute
from pytest import mark


class test_that_clamped_noise_does_not_go_below_floor:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        observation_matrix = to_array(np.eye(D_z := 3, D_x := 6))

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                original := noise.covariances(
                    process=1e-10,
                    observation=1e-10,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
            floor=(floor := noise.covariance_bounds(process=1e-5, observation=1e-5)),
        )

        model = provider(
            obstacle_count=(K := 1),
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(
                    mean=np.zeros((D_x, K)), covariance=np.eye(D_x)[:, :, np.newaxis]
                ),
                observation := np.zeros((D_z, K)),
                original,
                floor,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "original", "floor"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        original: NoiseT,
        floor: NoiseT,
    ) -> None:
        result, _ = model(
            noise=original,
            prediction=belief,
            observation=observation,
            state=model.state,
        )

        assert compute.min_eigenvalue(result.process_noise_covariance) >= floor.process
        assert (
            compute.min_eigenvalue(result.observation_noise_covariance)
            >= floor.observation
        )


class test_that_clamped_noise_does_not_go_above_ceiling:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        observation_matrix = to_array(np.eye(D_z := 3, D_x := 6))

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                original := noise.covariances(
                    process=10.0,
                    observation=10.0,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
            ceiling=(ceiling := noise.covariance_bounds(process=1.0, observation=1.0)),
        )

        model = provider(
            obstacle_count=(K := 1),
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(
                    mean=np.zeros((D_x, K)), covariance=np.eye(D_x)[:, :, np.newaxis]
                ),
                observation := np.zeros((D_z, K)),
                original,
                ceiling,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "original", "ceiling"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        original: NoiseT,
        ceiling: NoiseT,
    ) -> None:
        result, _ = model(
            noise=original,
            prediction=belief,
            observation=observation,
            state=model.state,
        )

        assert (
            compute.max_eigenvalue(result.process_noise_covariance) <= ceiling.process
        )
        assert (
            compute.max_eigenvalue(result.observation_noise_covariance)
            <= ceiling.observation
        )


class test_that_clamped_noise_is_not_changed_when_noise_is_above_floor_and_below_ceiling:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        observation_matrix = to_array(np.eye(D_z := 3, D_x := 6))

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                original := noise.covariances(
                    process=1.0,
                    observation=1.0,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
            floor=noise.covariance_bounds(process=1e-5, observation=1e-5),
            ceiling=noise.covariance_bounds(process=2.0, observation=2.0),
        )

        model = provider(
            obstacle_count=(K := 1),
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(
                    mean=np.zeros((D_x, K)), covariance=np.eye(D_x)[:, :, np.newaxis]
                ),
                observation := np.zeros((D_z, K)),
                original,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "original"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        original: NoiseT,
    ) -> None:
        result, _ = model(
            noise=original,
            prediction=belief,
            observation=observation,
            state=model.state,
        )

        assert np.all(
            result.process_noise_covariance == original.process_noise_covariance
        )
        assert np.all(
            result.observation_noise_covariance == original.observation_noise_covariance
        )


class test_that_noise_is_clamped_to_floor_and_ceiling_when_both_are_provided:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        observation_matrix = to_array(np.eye(D_z := 3, D_x := 6))

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                original := noise.covariances(
                    process=1e-10,
                    observation=20,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
            floor=(floor := noise.covariance_bounds(process=1e-5, observation=1e-5)),
            ceiling=(ceiling := noise.covariance_bounds(process=1.0, observation=1.0)),
        )

        model = provider(
            obstacle_count=(K := 1),
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(
                    mean=np.zeros((D_x, K)), covariance=np.eye(D_x)[:, :, np.newaxis]
                ),
                observation := np.zeros((D_z, K)),
                original,
                floor,
                ceiling,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "original", "floor", "ceiling"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        original: NoiseT,
        floor: NoiseT,
        ceiling: NoiseT,
    ) -> None:
        result, _ = model(
            noise=original,
            prediction=belief,
            observation=observation,
            state=model.state,
        )

        assert compute.min_eigenvalue(result.process_noise_covariance) >= floor.process
        assert (
            compute.min_eigenvalue(result.observation_noise_covariance)
            >= floor.observation
        )
        assert (
            compute.max_eigenvalue(result.process_noise_covariance) <= ceiling.process
        )
        assert (
            compute.max_eigenvalue(result.observation_noise_covariance)
            <= ceiling.observation
        )


class test_that_clamped_covariance_remains_symmetric_positive_definite:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        D_x, D_z = 6, 3
        observation_matrix = to_array(np.eye(D_z, D_x))

        def correlated_spd_matrix(
            *, dimension: int, variance: float, correlation: float
        ):
            return correlation * np.ones((dimension, dimension)) + (
                variance - correlation
            ) * np.eye(dimension)

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                original := noise.covariances(
                    process=to_array(
                        correlated_spd_matrix(
                            dimension=D_x, variance=5.0, correlation=4.0
                        )
                    ),
                    observation=to_array(
                        correlated_spd_matrix(
                            dimension=D_z, variance=0.5, correlation=0.1
                        )
                    ),
                )
            ),
            floor=noise.covariance_bounds(process=1.0, observation=1.0),
            ceiling=noise.covariance_bounds(process=2.0, observation=2.0),
        )

        model = provider(
            obstacle_count=(K := 1),
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(
                    mean=np.zeros((D_x, K)), covariance=np.eye(D_x)[:, :, np.newaxis]
                ),
                observation := np.zeros((D_z, K)),
                original,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "original"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        original: NoiseT,
    ) -> None:
        result, _ = model(
            noise=original,
            prediction=belief,
            observation=observation,
            state=model.state,
        )

        assert check.is_spd(np.asarray(result.process_noise_covariance))
        assert check.is_spd(np.asarray(result.observation_noise_covariance))
