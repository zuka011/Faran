# Risk Metrics

When obstacle positions are uncertain, a standard collision cost evaluates the expected (average) distance to obstacles. Risk metrics replace this expected value with a measure that is more sensitive to dangerous outcomes — the tail of the cost distribution.

Risk metrics are used in the [collision cost](costs.md#collision) to produce risk-aware trajectory planning. You configure them via the `metric` parameter.

## How Risk Metrics Work

Given $N$ samples of obstacle positions drawn from the predicted distribution:

1. The collision cost is evaluated for each sample independently.
2. The risk metric aggregates these $N$ cost values into a single scalar.
3. This scalar replaces the standard collision cost in the MPPI optimization.

Different metrics emphasize different parts of the cost distribution.

## Available Metrics

### Expected Value

The mean collision cost across all samples. Equivalent to not using a risk metric — it treats all outcomes equally.

```python
from faran.numpy import risk

metric = risk.expected_value(sample_count=50)
```

### Mean-Variance

Combines the mean and variance of the collision cost: $\rho = \mathbb{E}[J] + \gamma \, \text{Var}[J]$. Higher $\gamma$ penalizes cost variability, producing more conservative behavior.

```python
metric = risk.mean_variance(gamma=1.0, sample_count=50)
```

| Parameter | Effect |
|-----------|--------|
| $\gamma = 0$ | Pure expected value |
| $\gamma > 0$ | Penalizes high-variance outcomes; more cautious |

### Value at Risk (VaR)

The $\alpha$-quantile of the cost distribution: the cost below which $\alpha$% of outcomes fall. At $\alpha = 0.95$, VaR is the 95th-percentile cost.

```python
metric = risk.var(alpha=0.95, sample_count=50)
```

### Conditional Value at Risk (CVaR)

The expected cost in the worst $(1-\alpha)$% of outcomes. Also known as Expected Shortfall. More conservative than VaR because it averages the tail, not just the boundary.

$$
\text{CVaR}_\alpha = \mathbb{E}[J \mid J \geq \text{VaR}_\alpha]
$$

```python
metric = risk.cvar(alpha=0.95, sample_count=50)
```

**This is the most commonly used risk metric** for trajectory planning. At $\alpha = 0.95$, it considers the worst 5% of obstacle position samples.

### Entropic Risk

An exponential risk measure: $\rho = \frac{1}{\theta} \ln\!\left(\mathbb{E}[e^{\theta J}]\right)$. Positive $\theta$ is risk-averse; the magnitude controls sensitivity to tail costs.

```python
metric = risk.entropic_risk(theta=1.0, sample_count=50)
```

## Choosing a Metric

| Metric | Conservatism | Tuning | Notes |
|--------|-------------|--------|-------|
| Expected Value | Lowest | None | Risk-neutral baseline |
| Mean-Variance | Medium | $\gamma$ | Simple, easy to tune |
| VaR | Medium-High | $\alpha$ | Only considers the quantile boundary |
| **CVaR** | **High** | $\alpha$ | **Recommended default for safety-critical planning** |
| Entropic Risk | Adjustable | $\theta$ | Smooth, differentiable |

For most obstacle avoidance scenarios, start with **CVaR at $\alpha = 0.95$** and 50 samples.

## Usage with Collision Cost

Pass the metric to `costs.safety.collision`:

```python
from faran.numpy import costs, obstacles, risk
import numpy as np

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=np.array([0.5, 0.5, 0.5]),
    weight=1500.0,
    metric=risk.cvar(alpha=0.95, sample_count=50),
)
```

Without a `metric` argument, the collision cost uses deterministic evaluation (no sampling).

## Sample Count

The `sample_count` parameter controls how many obstacle position samples are drawn per evaluation. More samples give more accurate risk estimates but increase computation:

- **10–20 samples** — fast but noisy estimates
- **50 samples** — good balance for most scenarios
- **100+ samples** — smooth estimates, higher cost

## API Reference

See the [costs API reference](../api/costs.md) for collision cost signatures and the [feature overview](features.md) for the full list of supported risk metrics.
