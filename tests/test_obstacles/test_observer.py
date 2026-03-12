from typing import Sequence, Callable, Protocol
from dataclasses import dataclass, field

from faran import types, obstacles, ObstacleStatesForTimeStep, ObstacleStateObserver

from numtypes import array, Array

import numpy as np

from tests.dsl import mppi as data, check
from pytest import mark, Subtests


class StateWrapper(Protocol):
    def __call__(self, array: Array) -> ObstacleStatesForTimeStep:
        """Wraps a raw array representing obstacle states for a time step into the appropriate type."""
        ...


class ObstacleStateObserverProvider(Protocol):
    def __call__(
        self,
        inner: ObstacleStateObserver,
        *,
        sigma: Array,
        seed: int,
        to_states: Callable[[Array], ObstacleStatesForTimeStep],
    ) -> ObstacleStateObserver:
        """Creates a noisy obstacle state observer that decorates the given inner observer."""
        ...


@dataclass(frozen=True)
class ObstacleStateCollector:
    observed: list[ObstacleStatesForTimeStep] = field(default_factory=list)

    def observe(self, states: ObstacleStatesForTimeStep) -> None:
        self.observed.append(states)


class test_that_noisy_observer_delegates_modified_states_to_inner_observer:
    @staticmethod
    def cases(data, types, obstacles) -> Sequence[tuple]:
        return [
            (
                states := data.simple_obstacle_states_for_time_step(
                    array=array(
                        [
                            x := [0.0, 1.0, 2.0],
                            y := [2.0, 3.0, 4.0],
                            heading := [4.0, 3.0, 2.0],
                            velocity := [1.0, 0.5, 0.0],
                        ],
                        shape=(D_o := 4, K := 3),
                    )
                ),
                to_states := types.simple.obstacle_states_for_time_step.create,
                sigma := array([0.1, 0.2, 0.3, 0.4], shape=(D_o,)),
                create_observer := obstacles.observer.noisy,
            ),
            (
                states := data.obstacle_2d_poses_for_time_step(
                    x=array([0.0, 1.0, 2.0, 3.0], shape=(K := 4,)),
                    y=array([2.0, 3.0, 4.0, 5.0], shape=(K,)),
                    heading=array([4.0, 3.0, 2.0, 1.0], shape=(K,)),
                ),
                to_states := types.obstacle_2d_poses_for_time_step.wrap,
                sigma := array([0.1, 0.2, 0.3], shape=(3,)),
                create_observer := obstacles.observer.noisy,
            ),
        ]

    @mark.parametrize(
        ["states", "to_states", "sigma", "create_observer"],
        [
            *cases(data=data.numpy, types=types.numpy, obstacles=obstacles.numpy),
            *cases(data=data.jax, types=types.jax, obstacles=obstacles.jax),
        ],
    )
    def test(
        self,
        subtests: Subtests,
        states: ObstacleStatesForTimeStep,
        to_states: StateWrapper,
        sigma: Array,
        create_observer: ObstacleStateObserverProvider,
    ) -> None:
        observer = create_observer(
            inner := ObstacleStateCollector(), sigma=sigma, seed=42, to_states=to_states
        )

        for _ in range(1000):
            observer.observe(states)

        delegated = inner.observed

        with subtests.test("delegates exactly once per observation"):
            assert len(delegated) == 1000

        with subtests.test("delegated state has same shape as input states"):
            assert delegated[0].array.shape == states.array.shape

        with subtests.test("delegated state is different from input states"):
            assert not np.array_equal(delegated[0].array, states.array)

        with subtests.test("delegated state has approximately zero mean noise"):
            assert check.has_moments(
                samples=np.array([it.array - states.array for it in delegated]),
                mean=np.zeros_like(states.array),
                sigma=sigma,
            )


class test_that_states_are_unchanged_when_sigma_is_zero:
    @staticmethod
    def cases(data, types, obstacles) -> Sequence[tuple]:
        return [
            (
                data.simple_obstacle_states_for_time_step(
                    array=array(
                        [
                            [0.0, 1.0, 2.0],
                            [2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0],
                            [1.0, 0.5, 0.0],
                        ],
                        shape=(D_o := 4, K := 3),
                    )
                ),
                types.simple.obstacle_states_for_time_step.create,
                array([0.0, 0.0, 0.0, 0.0], shape=(D_o,)),
                create_observer := obstacles.observer.noisy,
            ),
            (
                data.obstacle_2d_poses_for_time_step(
                    x=array([0.0, 1.0, 2.0, 3.0], shape=(K := 4,)),
                    y=array([2.0, 3.0, 4.0, 5.0], shape=(K,)),
                    heading=array([4.0, 3.0, 2.0, 1.0], shape=(K,)),
                ),
                types.obstacle_2d_poses_for_time_step.wrap,
                array([0.0, 0.0, 0.0], shape=(3,)),
                create_observer := obstacles.observer.noisy,
            ),
        ]

    @mark.parametrize(
        ["states", "to_states", "sigma", "create_observer"],
        [
            *cases(data=data.numpy, types=types.numpy, obstacles=obstacles.numpy),
            *cases(data=data.jax, types=types.jax, obstacles=obstacles.jax),
        ],
    )
    def test(
        self,
        states: ObstacleStatesForTimeStep,
        to_states: StateWrapper,
        sigma: Array,
        create_observer: ObstacleStateObserverProvider,
    ) -> None:
        observer = create_observer(
            inner := ObstacleStateCollector(), sigma=sigma, seed=42, to_states=to_states
        )

        observer.observe(states)

        assert np.array_equal(inner.observed[0].array, states.array)


class test_that_same_seed_produces_identical_noise:
    @staticmethod
    def cases(data, types, obstacles) -> Sequence[tuple]:
        return [
            (
                data.simple_obstacle_states_for_time_step(
                    array=array(
                        [
                            [0.0, 1.0, 2.0],
                            [2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0],
                            [1.0, 0.5, 0.0],
                        ],
                        shape=(D_o := 4, K := 3),
                    )
                ),
                types.simple.obstacle_states_for_time_step.create,
                array([0.1, 0.2, 0.3, 0.4], shape=(D_o,)),
                create_observer := obstacles.observer.noisy,
            ),
            (
                data.obstacle_2d_poses_for_time_step(
                    x=array([0.0, 1.0, 2.0, 3.0], shape=(K := 4,)),
                    y=array([2.0, 3.0, 4.0, 5.0], shape=(K,)),
                    heading=array([4.0, 3.0, 2.0, 1.0], shape=(K,)),
                ),
                types.obstacle_2d_poses_for_time_step.wrap,
                array([0.1, 0.2, 0.3], shape=(3,)),
                create_observer := obstacles.observer.noisy,
            ),
        ]

    @mark.parametrize(
        ["states", "to_states", "sigma", "create_observer"],
        [
            *cases(data=data.numpy, types=types.numpy, obstacles=obstacles.numpy),
            *cases(data=data.jax, types=types.jax, obstacles=obstacles.jax),
        ],
    )
    def test(
        self,
        states: ObstacleStatesForTimeStep,
        to_states: StateWrapper,
        sigma: Array,
        create_observer: ObstacleStateObserverProvider,
    ) -> None:
        observer_a = create_observer(
            inner_a := ObstacleStateCollector(),
            sigma=sigma,
            seed=42,
            to_states=to_states,
        )
        observer_b = create_observer(
            inner_b := ObstacleStateCollector(),
            sigma=sigma,
            seed=42,
            to_states=to_states,
        )

        observer_a.observe(states)
        observer_b.observe(states)

        assert np.array_equal(inner_a.observed[0].array, inner_b.observed[0].array)


class test_that_different_seeds_produce_different_noise:
    @staticmethod
    def cases(data, types, obstacles) -> Sequence[tuple]:
        return [
            (
                data.simple_obstacle_states_for_time_step(
                    array=array(
                        [
                            [0.0, 1.0, 2.0],
                            [2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0],
                            [1.0, 0.5, 0.0],
                        ],
                        shape=(D_o := 4, K := 3),
                    )
                ),
                types.simple.obstacle_states_for_time_step.create,
                array([0.1, 0.2, 0.3, 0.4], shape=(D_o := 4,)),
                create_observer := obstacles.observer.noisy,
            ),
            (
                data.obstacle_2d_poses_for_time_step(
                    x=array([0.0, 1.0, 2.0, 3.0], shape=(K := 4,)),
                    y=array([2.0, 3.0, 4.0, 5.0], shape=(K,)),
                    heading=array([4.0, 3.0, 2.0, 1.0], shape=(K,)),
                ),
                types.obstacle_2d_poses_for_time_step.wrap,
                array([0.1, 0.2, 0.3], shape=(3,)),
                create_observer := obstacles.observer.noisy,
            ),
        ]

    @mark.parametrize(
        ["states", "to_states", "sigma", "create_observer"],
        [
            *cases(data=data.numpy, types=types.numpy, obstacles=obstacles.numpy),
            *cases(data=data.jax, types=types.jax, obstacles=obstacles.jax),
        ],
    )
    def test(
        self,
        states: ObstacleStatesForTimeStep,
        to_states: StateWrapper,
        sigma: Array,
        create_observer: ObstacleStateObserverProvider,
    ) -> None:
        observer_a = create_observer(
            inner_a := ObstacleStateCollector(),
            sigma=sigma,
            seed=42,
            to_states=to_states,
        )
        observer_b = create_observer(
            inner_b := ObstacleStateCollector(),
            sigma=sigma,
            seed=99,
            to_states=to_states,
        )

        observer_a.observe(states)
        observer_b.observe(states)

        assert not np.array_equal(inner_a.observed[0].array, inner_b.observed[0].array)


class test_that_original_states_are_not_mutated:
    @staticmethod
    def cases(data, types, obstacles) -> Sequence[tuple]:
        return [
            (
                data.simple_obstacle_states_for_time_step(
                    array=array(
                        [
                            [0.0, 1.0, 2.0],
                            [2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0],
                            [1.0, 0.5, 0.0],
                        ],
                        shape=(D_o := 4, K := 3),
                    )
                ),
                types.simple.obstacle_states_for_time_step.create,
                array([0.5, 0.5, 0.5, 0.5], shape=(D_o := 4,)),
                create_observer := obstacles.observer.noisy,
            ),
            (
                data.obstacle_2d_poses_for_time_step(
                    x=array([0.0, 1.0, 2.0, 3.0], shape=(K := 4,)),
                    y=array([2.0, 3.0, 4.0, 5.0], shape=(K,)),
                    heading=array([4.0, 3.0, 2.0, 1.0], shape=(K,)),
                ),
                types.obstacle_2d_poses_for_time_step.wrap,
                array([0.5, 0.5, 0.5], shape=(3,)),
                create_observer := obstacles.observer.noisy,
            ),
        ]

    @mark.parametrize(
        ["states", "to_states", "sigma", "create_observer"],
        [
            *cases(data=data.numpy, types=types.numpy, obstacles=obstacles.numpy),
            *cases(data=data.jax, types=types.jax, obstacles=obstacles.jax),
        ],
    )
    def test(
        self,
        states: ObstacleStatesForTimeStep,
        to_states: StateWrapper,
        sigma: Array,
        create_observer: ObstacleStateObserverProvider,
    ) -> None:
        original_array = states.array.copy()
        observer = create_observer(
            ObstacleStateCollector(), sigma=sigma, seed=42, to_states=to_states
        )

        observer.observe(states)

        assert np.array_equal(states.array, original_array)
