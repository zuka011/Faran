---
status: draft
---

# obstacles

Obstacle avoidance pipeline: state observation, distance computation, and collision cost evaluation. When obstacle positions are uncertain, a [risk metric](../costs/risk.md) replaces the deterministic collision cost with risk-sensitive evaluation.

## Pipeline

```
Observation → Distance computation → Collision cost
```

## Factory Namespace

| Factory | Description |
|---------|-------------|
| `obstacles.static(...)` | Static obstacle simulator |
| `obstacles.dynamic(...)` | Constant-velocity obstacle simulator |
| `obstacles.empty()` | Empty simulator (zero obstacles) |
| `obstacles.provider.predicting(...)` | Observation + prediction provider |
| `obstacles.sampler.gaussian(...)` | Gaussian pose sampler from predicted covariances |
| `obstacles.observer.noisy(...)` | Adds Gaussian noise to observations |
| `obstacles.id_assignment.hungarian(...)` | Hungarian-algorithm ID matching |

## Distance Computation

Two methods compute signed distances between the ego vehicle and obstacles. Negative values indicate overlap.

### Circle-to-Circle

Represents both the ego and obstacles as collections of circles[@Tolksdorf2024]:

```python
from faran.numpy import distance, extract, obstacles
from faran import Circles
from numtypes import array

distance_extractor = distance.circles(
    ego=Circles(
        origins=array([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]], shape=(3, 2)),
        radii=array([0.8, 0.8, 0.8], shape=(3,)),
    ),
    obstacle=Circles(
        origins=array([[0.0, 0.0]], shape=(1, 2)),
        radii=array([1.0], shape=(1,)),
    ),
    position_extractor=extract.from_physical(lambda states: states.positions),
    heading_extractor=extract.from_physical(heading),
    obstacle_position_extractor=obstacles.pose_position_extractor,
    obstacle_heading_extractor=obstacles.pose_heading_extractor,
)
```

### SAT (Separating Axis Theorem)

Represents both the ego and obstacles as convex polygons[@Gottschalk1997]:

```python
from faran import ConvexPolygon

distance_extractor = distance.sat(
    ego=ConvexPolygon.rectangle(length=2.5, width=1.2),
    obstacle=ConvexPolygon.rectangle(length=2.5, width=1.2),
    position_extractor=extract.from_physical(lambda states: states.positions),
    heading_extractor=extract.from_physical(heading),
    obstacle_position_extractor=obstacles.pose_position_extractor,
    obstacle_heading_extractor=obstacles.pose_heading_extractor,
)
```

## Collision Cost

Penalizes rollouts where signed distance drops below a threshold. See [Safety Costs](../costs/safety.md) for the collision cost and [Risk Metrics](../costs/risk.md) for risk-aware evaluation.

```python
from faran.numpy import costs, obstacles
from numtypes import array

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
)
```

## State Provider

::: faran.types.ObstacleStateProvider
    options:
      show_root_heading: true
      heading_level: 3

## Obstacle Pose Sampler

Draws samples from predicted obstacle state distributions parameterized by covariance matrices. Used for risk-aware collision cost evaluation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed |

::: faran.obstacles.sampler.basic.NumPyGaussianObstacle2dPoseSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.obstacles.sampler.accelerated.JaxGaussianObstacle2dPoseSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

\bibliography
