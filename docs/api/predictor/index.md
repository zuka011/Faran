---
status: draft
---

# predictor

Obstacle motion predictors estimate future obstacle states. Two predictors are available:

| Predictor | Factory | Description |
|-----------|---------|-------------|
| Curvilinear | `predictor.curvilinear(...)` | Forward-propagates estimated states with a motion model and Kalman filter |
| Static | `predictor.static(...)` | Replicates the last observed state over the horizon |

## Curvilinear Predictor

Forward-propagates obstacle states using a [motion model](../model/index.md) and [state estimator](../estimation/index.md). With Kalman filter-based estimators, covariance is propagated automatically[@Schubert2008].

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizon` | `int` | — | Prediction horizon $T$ |
| `model` | `ObstacleModel` | — | Motion model for forward propagation |
| `estimator` | `ObstacleStateEstimator` | — | Estimates current state and inputs from history |
| `prediction` | `PredictionCreator` | — | Converts propagated state sequences to obstacle states |
| `assumptions` | `InputAssumptionProvider \| None` | `None` | Overrides estimated inputs before propagation. Defaults to identity (no change). |

### Usage

```python
from faran.numpy import predictor, model, obstacles, types

motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
    estimator=model.bicycle.estimator.ekf(
        time_step_size=0.1, wheelbase=2.5,
        process_noise_covariance=1e-3, observation_noise_covariance=1e-2,
    ),
    prediction=bicycle_to_obstacle_states,
)

provider = obstacles.provider.predicting(
    predictor=motion_predictor,
    history=types.obstacle_states_running_history.empty(horizon=2),
)
```

## Static Predictor

Replicates the last observed state over the prediction horizon.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `horizon` | `int` | Prediction horizon $T$ |

## Predicting Obstacle State Provider

Combines a predictor with a running state history. Observations are appended via `observe()`, and predictions are retrieved by calling the provider.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictor` | `ObstacleMotionPredictor` | — | Predictor to use |
| `history` | `ObstacleStatesRunningHistory` | — | Running observation history |
| `id_assignment` | `ObstacleIdAssignment \| None` | `None` | ID matching strategy. Defaults to no-op. |

::: faran.obstacles.PredictingObstacleStateProvider
    options:
      show_root_heading: true
      heading_level: 3

## Protocols

::: faran.types.ObstacleMotionPredictor
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.InputAssumptionProvider
    options:
      show_root_heading: true
      heading_level: 3

\bibliography
