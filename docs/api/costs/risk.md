---
status: draft
---

# Risk Metrics

Risk metrics aggregate sampled collision costs into a single scalar that emphasizes dangerous tail outcomes. Passed to [collision cost](safety.md#collision-cost) via the `metric` parameter.

## Available Metrics

| Metric | Formula | Parameters |
|--------|---------|------------|
| Expected Value | $\mathbb{E}[J]$ | `sample_count` |
| Mean-Variance | $\mathbb{E}[J] + \gamma \, \text{Var}[J]$ | `gamma`, `sample_count` |
| VaR | $\inf\{c : P(J \leq c) \geq \alpha\}$ | `alpha`, `sample_count` |
| CVaR | $\mathbb{E}[J \mid J \geq \text{VaR}_\alpha]$ | `alpha`, `sample_count` |
| Entropic | $\frac{1}{\theta}\ln\!\bigl(\mathbb{E}[e^{\theta J}]\bigr)$ | `theta`, `sample_count` |

### Expected Value

Risk-neutral baseline. Equivalent to no risk metric.

```python
from faran.numpy import risk

metric = risk.expected_value(sample_count=50)
```

### Mean-Variance

$\gamma = 0$ recovers expected value; $\gamma > 0$ penalizes cost variability.

```python
metric = risk.mean_variance(gamma=1.0, sample_count=50)
```

### Value at Risk

$\alpha$-quantile of the cost distribution.

```python
metric = risk.var(alpha=0.95, sample_count=50)
```

### Conditional Value at Risk

Expected cost in the worst $(1 - \alpha)$ fraction. Recommended default for safety-critical planning.

```python
metric = risk.cvar(alpha=0.95, sample_count=50)
```

### Entropic Risk

Positive $\theta$ is risk-averse; magnitude controls tail sensitivity.

```python
metric = risk.entropic_risk(theta=1.0, sample_count=50)
```

### No Metric

Deterministic single-sample evaluation. Used when `metric` is omitted from the collision cost.

```python
metric = risk.none()
```

## Usage

```python
from faran.numpy import costs, risk
import numpy as np

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacle_sampler,
    distance=distance_extractor,
    distance_threshold=np.array([0.5, 0.5, 0.5]),
    weight=1500.0,
    metric=risk.cvar(alpha=0.95, sample_count=50),
)
```

::: faran.costs.risk.base.RisKitRiskMetric
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - create
