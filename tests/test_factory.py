from typing import Sequence, Callable

from faran import MpccMppiSetup, mppi, model, sampler, trajectory, types, extract

from numtypes import array

from pytest import mark


class test_that_mpcc_factory_creates_mpcc_planner:
    @staticmethod
    def cases(mppi, model, sampler, trajectory, types) -> Sequence[tuple]:
        def position(states):
            return types.positions(x=states.positions.x(), y=states.positions.y())

        reference = trajectory.waypoints(
            points=array([[0, 0], [10, 0], [20, 5], [30, 5]], shape=(4, 2)),
            path_length=35.0,
        )

        return [
            lambda: mppi.mpcc(
                model=model.bicycle.dynamical(
                    time_step_size=0.1,
                    wheelbase=2.5,
                    speed_limits=(0.0, 15.0),
                    steering_limits=(-0.5, 0.5),
                    acceleration_limits=(-3.0, 3.0),
                ),
                sampler=sampler.gaussian(
                    standard_deviation=array([0.5, 0.2], shape=(2,)),
                    rollout_count=256,
                    to_batch=types.bicycle.control_input_batch.create,
                    seed=42,
                ),
                reference=reference,
                position_extractor=extract.from_physical(position),
                config={
                    "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
                    "virtual": {"velocity_limits": (0.0, 15.0)},
                },
            )
        ]

    @mark.parametrize(
        "factory",
        [
            *cases(
                mppi=mppi.numpy,
                model=model.numpy,
                sampler=sampler.numpy,
                trajectory=trajectory.numpy,
                types=types.numpy,
            ),
            *cases(
                mppi=mppi.jax,
                model=model.jax,
                sampler=sampler.jax,
                trajectory=trajectory.jax,
                types=types.jax,
            ),
        ],
    )
    def test(self, factory: Callable[[], MpccMppiSetup]) -> None:
        setup = factory()

        assert setup.mppi is not None
        assert setup.model is not None
        assert setup.contouring_cost is not None
        assert setup.lag_cost is not None
