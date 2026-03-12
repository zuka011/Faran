# Motion Prediction

The obstacle state provider supplies predicted obstacle positions to the collision cost at each planning step. It wraps a motion predictor and maintains a running history of observations.

## Curvilinear Predictor

Propagates obstacle states forward using constant-input motion models[@Schubert2008] (constant velocity, constant steering angle and acceleration, etc.):

```python
from faran.numpy import obstacles, predictor, model, types

motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
    prediction=bicycle_to_obstacle_states,
)

provider = obstacles.provider.predicting(
    predictor=motion_predictor,
    history=types.obstacle_states_running_history.empty(horizon=2),
)

# Feed observations each step
provider.observe(detected_obstacle_states)
```

## Input Assumptions

By default the curvilinear predictor assumes all estimated input components remain constant over the horizon. Override specific components with an `assumptions` callable:

```python
from faran.models.bicycle.basic import NumPyBicycleObstacleInputs
import numpy as np

# Keep acceleration, zero out steering angle
motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
    prediction=bicycle_to_obstacle_states,
    assumptions=lambda v: NumPyBicycleObstacleInputs(
        accelerations=v.accelerations,
        steering_angles=np.zeros_like(v.steering_angles),
    ),
)
```

```python
from faran.models.unicycle.basic import NumPyUnicycleObstacleInputs

# Keep linear velocity, zero out angular velocity
motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.unicycle.obstacle(time_step_size=0.1),
    prediction=unicycle_to_obstacle_states,
    assumptions=lambda v: NumPyUnicycleObstacleInputs(
        linear_velocities=v.linear_velocities,
        angular_velocities=np.zeros_like(v.angular_velocities),
    ),
)
```

## Noisy Observations

For testing or estimator tuning, inject zero-mean Gaussian noise into obstacle detections:

```python
from faran.numpy import noise
from numtypes import array

observer = noise.adaptive(
    provider,
    to_states=types.obstacle_2d_poses_for_time_step.wrap,
    sigma=array([0.1, 0.1, 0.05], shape=(3,)),
    seed=44,
)

observer.observe(detected_obstacle_states)
```

## API Reference

See the [predictor API reference](../../api/predictor.md) for full signatures.

\bibliography
