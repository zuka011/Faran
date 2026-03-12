from typing import Final


from faran.obstacles.common import PredictingObstacleStateProvider
from faran.obstacles.static import (
    NumPyStaticObstacleSimulator,
    JaxStaticObstacleSimulator,
)
from faran.obstacles.dynamic import (
    NumPyDynamicObstacleSimulator,
    JaxDynamicObstacleSimulator,
)
from faran.obstacles.sampler import (
    NumPyGaussianObstacle2dPoseSampler,
    JaxGaussianObstacle2dPoseSampler,
)
from faran.obstacles.assignment import (
    NumPyHungarianObstacleIdAssignment,
    JaxHungarianObstacleIdAssignment,
)
from faran.obstacles.observer import NoisyObstacleStateObserver


class obstacles:
    class numpy:
        empty: Final = NumPyStaticObstacleSimulator.empty
        static: Final = NumPyStaticObstacleSimulator.create
        dynamic: Final = NumPyDynamicObstacleSimulator.create

        class provider:
            predicting: Final = PredictingObstacleStateProvider.create

        class sampler:
            gaussian: Final = NumPyGaussianObstacle2dPoseSampler.create

        class id_assignment:
            hungarian: Final = NumPyHungarianObstacleIdAssignment.create

        class observer:
            noisy: Final = NoisyObstacleStateObserver.decorate

    class jax:
        empty: Final = JaxStaticObstacleSimulator.empty
        static: Final = JaxStaticObstacleSimulator.create
        dynamic: Final = JaxDynamicObstacleSimulator.create

        class provider:
            predicting: Final = PredictingObstacleStateProvider.create

        class sampler:
            gaussian: Final = JaxGaussianObstacle2dPoseSampler.create

        class id_assignment:
            hungarian: Final = JaxHungarianObstacleIdAssignment.create

        class observer:
            noisy: Final = NoisyObstacleStateObserver.decorate
