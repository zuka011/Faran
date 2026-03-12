# Distance Computation

Two methods compute signed distances between the ego vehicle and obstacles.

## Circle-to-Circle

Represents both the ego and obstacles as collections of circles[@Tolksdorf2024]. Distance is center-to-center minus both radii. Negative values indicate overlap.

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
    position_extractor=extract.from_physical(
        lambda states: states.positions
    ),
    heading_extractor=extract.from_physical(heading),
    obstacle_position_extractor=obstacles.pose_position_extractor,
    obstacle_heading_extractor=obstacles.pose_heading_extractor,
)
```

Fast, suitable when precise geometry is not needed.

## SAT (Separating Axis Theorem)

Represents both the ego and obstacles as convex polygons[@Gottschalk1997]. Computes exact signed separation distance.

```python
from faran import ConvexPolygon

distance_extractor = distance.sat(
    ego=ConvexPolygon.rectangle(length=2.5, width=1.2),
    obstacle=ConvexPolygon.rectangle(length=2.5, width=1.2),
    position_extractor=extract.from_physical(
        lambda states: states.positions
    ),
    heading_extractor=extract.from_physical(heading),
    obstacle_position_extractor=obstacles.pose_position_extractor,
    obstacle_heading_extractor=obstacles.pose_heading_extractor,
)
```

More accurate but limited to 2D convex shapes.

## API Reference

See the [obstacles API reference](../../api/obstacles.md) for full signatures.

\bibliography
