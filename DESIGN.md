# Design Guide

This document describes the design principles, code style, and testing philosophy used in Faran. It's intended as a reference for contributors who want to understand the "why" behind our conventions. It's important to note that there could exist cases where deviating from these guidelines is justified, so use your judgement and feel free to discuss any such cases in an MR or issue.

This is **not** a checklist of rules to follow, so you don't need to read the whole thing! See the [contributing guide](CONTRIBUTING.md) if you just want to get started with contributing. Here's a quick summary of what you can find in this document:

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Code Style](#code-style)
  - [General Principles](#general-principles)
- [Testing Philosophy](#testing-philosophy)
  - [Testing as a Design Tool](#testing-as-a-design-tool)
  - [Test Naming Convention](#test-naming-convention)
  - [Test Structure](#test-structure)
  - [Parametrized Tests & Testing Multiple Backends](#parametrized-tests--testing-multiple-backends)
  - [Subtests](#subtests)
  - [Test DSL](#test-dsl)
  - [Test Doubles](#test-doubles)
  - [Testing Gotchas](#testing-gotchas)
    - [Walrus Operator and Lambda Functions](#walrus-operator-and-lambda-functions)
  - [Testing Numerical Code](#testing-numerical-code)
    - [Don't Reimplement the Algorithm in the Test](#dont-reimplement-the-algorithm-in-the-test)
    - [Try to Test Mathematical Properties First](#try-to-test-mathematical-properties-first)
    - [Decompose Algorithms into Smaller, Testable Pieces](#decompose-algorithms-into-smaller-testable-pieces)
  - [Docstrings](#docstrings)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Code Style

### General Principles

1. **Decomposition over comments**: Use well-named functions and classes to explain your code rather than comments. AI assistants (e.g. GitHub Copilot, Claude Code) tend to generate verbosely commented and badly structured code, so be sure to pay extra attention to this if you do use AI assistance. You can, of course, still use a comment if there is no better way to give a developer necessary "why" context about a decision.

2. **Descriptive names**: Use full words in identifiers. You may omit words when context is clear, but never abbreviate.
   ```python
    # Good
    control_input_sequence = ...
    rollout_count = ...
    
    # Bad
    ctrl_inp_seq = ...
    n_rollouts = ...
   ```
    The minor convenience of shorter names is not worth the cognitive overhead of deciphering abbreviations, especially for those unfamiliar with the codebase. When dealing with mathematical code, common symbols can be used as variable names, but only when their meaning is unambiguous in context. This is usually never the case for public APIs though, so even if `dt` is obviously the time step size, it's still better to name a factory parameter `time_step_size` instead.

3. **Immutability**: Prefer immutable data structures, e.g. frozen dataclasses or namedtuples:
   ```python
    # Good
    @dataclass(frozen=True)
    class BicycleState:
        x: float
        y: float
        heading: float
        speed: float

    # Also good
    class BicycleState(NamedTuple):
         x: float
         y: float
         heading: float
         speed: float

    # Bad, and verbose
    class BicycleState:
        def __init__(self, x: float, y: float, heading: float, speed: float):
            self.x = x
            self.y = y
            self.heading = heading
            self.speed = speed
   ```
    State makes components harder to test and reason about. Also, most Faran components are designed to be stateless anyways (including planners).

4. **Functional style**: Prefer declarative/functional patterns over imperative loops. The reasoning is similar to the previous point about immutability. Additionally, such a style makes porting implementations of algorithms to some backends easier (e.g. JAX).

5. **Static typing**: All code must be fully typed. We use:
   - `pyright` for type checking
   - `beartype` for runtime validation
   - `jaxtyping` for array shape annotations
   
   In reasonable cases, you may omit type annotations (e.g. in tests) or use `Any`, `# type: ignore` when the static typing is doing more harm than good.

## Testing Philosophy

### Testing as a Design Tool

Faran was designed with a test-driven approach. Here's a short blueprint explaining what that means.

1. When a feature is desired, the first question to ask is "What would the ideal API for this feature look like?". The answer to that question may be complicated, so start with the simplest use case you can think of and try to capture it with a code snippet. 
2. Once you have a more-or-less concrete understanding of (a part of) the feature, you can formulate a test to also explicitly state the expected behavior. Try to hide as much technical detail as possible from the test (e.g. underlying data representation, numerical backend).
3. Implement the simplest solution that makes your test pass. At this point you have the choice to either iterate on this test and API to make it "cleaner", or proceed to the next (part of the) feature.

### Test Naming Convention

Tests follow this naming pattern:

```
test_that_<functionality>[_when_<condition>]
```

Examples:
- `test_that_mppi_favors_samples_with_lower_costs`
- `test_that_tracking_cost_does_not_depend_on_coordinate_system`
- `test_that_query_returns_first_waypoint_when_path_parameter_is_zero`

### Test Structure

Tests typically use the **Arrange-Act-Assert** pattern:

```python
def test_that_mppi_favors_samples_with_lower_costs():
    # Arrange: Set up test data
    mppi = create_mppi(...)
    temperature = 0.1
    nominal_input = ...
    initial_state = ...
    expected = ...
    tolerance = 1e-3

    # Act: Call the function under test
    result = mppi.step(
        temperature=temperature,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )
    
    # Assert: Check the results
    assert np.allclose(result.optimal.array, expected.array, atol=tolerance)
```

### Parametrized Tests & Testing Multiple Backends

Sometimes we use parameterized tests to run the same logic against multiple components. If the parameterization logic is complicated, or the test case generation itself needs to be parameterized (e.g. to run against components with different backends), we organize the test as a class instead:

```python
class test_that_mppi_favors_samples_with_lower_costs:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                # Arrange: Set up test data agnostic to backend
                mppi := create_mppi.base(...),
                temperature := 0.1,
                nominal_input,
                initial_state,
                expected := ...,
                tolerance := 1e-3,
            ),
        ]

    @mark.parametrize(
        ["mppi", "temperature", "nominal_input", "initial_state", "expected", "tolerance"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test(
        self, 
        mppi: Mppi, 
        temperature: float, 
        nominal_input: ControlInputSequence, 
        initial_state: State, 
        expected: ControlInputSequence, 
        tolerance: float
    ) -> None:
        # Act
        result = mppi.step(
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        
        # Assert
        assert np.allclose(result.optimal.array, expected.array, atol=tolerance)
```

The `cases` static method has the additional benefit that it is scoped to the test class (does not need the test name as a qualifier) and does not pollute the module namespace. When there are a lot of test cases like this, the extra bit of organization is really helpful. Also see [Testing Gotchas](#testing-gotchas) for some parameterization pitfalls.

All functionality must work identically on all backends. Using parameterized tests like this also ensures components implemented using different backends have similar behavior (ignoring small numerical differences) and a common API.

```python
class test_that_...:
    @mark.parametrize(
        ["trajectory", "expected"],
        [
            *cases(trajectory=trajectory.numpy),
            *cases(trajectory=trajectory.jax),
        ],
    )
    def test(self, trajectory, expected):
        ...
```

### Subtests

If you have a complex setup, but want to check multiple different behaviors/properties of a component, you can use subtests:

```python
from pytest import SubTests

def test_that_prediction_error_covariance_is_positive_definite(subtests: SubTests) -> None:
    # Arrange
    model = create_model(...)
    inputs = ...
    initial_state = ...
    
    # Act
    result = model.predict(inputs=inputs, initial_state=initial_state)
    
    # Assert
    with subtests.test("Covariance is symmetric"):
        assert np.allclose(result.covariance, result.covariance.T)

    with subtests.test("Covariance is positive definite"):
        assert np.all(np.linalg.eigvals(result.covariance) > 0)
```

### Test DSL

If your test setup is complicated, consider creating some simple DSL to express your intent in a more readable way.

```python
# Good
from tests.dsl import check

result = model.predict(...)
check.is_spd(result.covariance, atol=1e-8)
    
# Bad
result = model.predict(...)

for obstacle_covariance in result.covariances:
    assert np.allclose(obstacle_covariance, obstacle_covariance.T, atol=1e-8)
    assert np.all(np.linalg.eigvals(obstacle_covariance) > 0)
```

### Test Doubles

**Do not use mocks.** Use stubs, fake implementations or real components instead. This helps keep the tests black-box and focused on the behavior of the component under test, rather than its internal implementation. It also makes the tests more robust to refactoring and implementation changes[<sup>1</sup>](https://martinfowler.com/articles/mocksArentStubs.html).

```python
from tests.dsl import stubs

# Good: Stub that returns predetermined values
model = stubs.DynamicalModel.returns(
    rollouts=expected_rollouts,
    when_control_inputs_are=inputs,
    and_initial_state_is=initial_state,
)
planner = Mppi(..., model=model)

# Good: Fake component that implements the same interface but with toy logic
model = SimpleLinearModel(...)
planner = Mppi(..., model=model)

# Bad: Mocking the model's behavior and hardcoding its specs into the test. The 
# specs of the DynamicalModel are implementation details in the context of this test, 
# since the SUT is the MPPI planner, not the model.
from unittest.mock import Mock

model = Mock(spec=DynamicalModel)
model.predict.return_value = expected_rollouts

planner = Mppi(..., model=model)
```

### Testing Gotchas

#### Walrus Operator and Lambda Functions

In the test case tuples, we use the walrus operator (`variable := value`) to make it obvious what each element represents. However, be careful when combining this with lambda functions. The walrus operator assigns in the enclosing scope, so if you reuse the same variable name across tuples, you're overwriting it. Lambdas then look up that variable at call time, so they all see the [final value](https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result):

```python
# E.g. if these are your cases.
cases = [
    (a := 1, f := lambda: a),
    (a := 2, f := lambda: a),
]

# Then you're going to get these results.
assert cases[0][1]() == 2  # That's not a 1...
assert cases[1][1]() == 2
```

The fix is usually straightforward. Just use default arguments in the lambda to capture the value at definition time:

```python
cases = [
    (a := 1, f := lambda a=a: a),
    (a := 2, f := lambda a=a: a),
]

# This is what you would expect.
assert cases[0][1]() == 1
assert cases[1][1]() == 2
```

That now works as intended. Sometimes using the walrus operator can also cause type checking issues, so you may still need to use a different variable name.

### Testing Numerical Code

Testing numerical code generally offers many opportunities to mess up and write fragile, flaky, and overall weak tests. Here are a few tips to help you keep the test suite maintainable and actually helpful.

#### Don't Reimplement the Algorithm in the Test

Say you want to implement a function to find the roots of a quadratic polynomial of the form $a x^2 + b x + c = 0$. In code, that would look like this:

```python
def roots_of_quadratic_polynomial(*, a: float, b: float, c: float) -> tuple[float, float]:
    # TODO: Implement this!
    pass

x_1, x_2 = roots_of_quadratic_polynomial(a=1, b=0, c=-1)
```

It may be tempting to write something like this in the test:

```python
@mark.parameterize(
    ["a", "b", "c"],
    [
        # Simple case where the roots have 0 offset along the x axis.
        (a := 1, b := 0, c := 1)
    ]
)
def test_that_roots_of_quadratic_polynomials_are_correct(a: float, b: float, c: float) -> None:
    D = b**2 - 4 * a * c
    expected = (-b - sqrt(D)) / (2 * a), (-b + sqrt(D)) / (2 * a)

    assert roots_of_quadratic_polynomial(a=a, b=b, c=c) == approx(expected)
```

However, if you then go ahead and implement this function using the same analytical solution, you're really just verifying that ```1 == 1```.

> **NOTE:** It would make sense to write the above test if you use then proceed to implement the root finding function using Newton's method or in some other way that is more complex than what's written in the test.

A better test would look as follows:

```python
@mark.parameterize(
    ["a", "b", "c", "expected"],
    [
        # Simple case where the roots have 0 offset along the x axis. Thus, we know the roots are at +1 and -1.
        (a := 1, b := 0, c := 1, expected := (-1, +1))
        # More tests for simple cases, where the root is known, or can be computed trivially...
    ]
)
def test_that_roots_of_quadratic_polynomials_are_correct(a: float, b: float, c: float, expected: tuple[float, float]) -> None:
    assert roots_of_quadratic_polynomial(a=a, b=b, c=c) == approx(expected)
```

#### Try to Test Mathematical Properties First 

We can make the above test even stronger and easier to write/maintain by focusing on what properties we want satisfied, instead of specific results.

A root of a function $f(x)$ is by definition a point $x_0$ in the function's domain for which the function value is zero, i.e. $f(x=x_0) = 0$. Well, that's not too hard to test:

```python
@mark.parameterize(
    ["a", "b", "c"],
    [
        # I have no idea what the answer here should be.
        (a := 1, b := 2, c := -3)
        # But that's okay, since the function value at the root should always be zero, regardless of what the root is.
        (a := 6, b := -4, c := -42)
        # Be careful though, some properties may not be satisfied in all circumstances. For example, this function has no root.
        (a := 0, b := 0, c := 5)
    ]
)
def test_that_value_of_quadratic_polynomial_is_zero_when_argument_is_a_root(a: float, b: float, c: float) -> None:
    x_1, x_2 = roots_of_quadratic_polynomial(a=a, b=b, c=c)

    assert a * x_1**2 + b * x_1 + c == approx(0.0)
    assert a * x_2**2 + b * x_2 + c == approx(0.0)
```

In this case, the test is simply easier to write and less error-prone. In other cases, focusing on properties can make the tests more robust too and even be reused for multiple algorithms that try to do the same thing.

#### Decompose Algorithms into Smaller, Testable Pieces

If the algorithm under test has multiple steps, it can sometimes be helpful to abstract away some "difficult" steps in a way that makes testing easier. Non-deterministic steps (like sampling) are usually the first candidates for this. 

Say you have an ensemble of neural networks and you want to compute the disagreement among the ensemble members' predictions as a measure of epistemic uncertainty. You can do this by computing the trace of the covariance matrix of the predictions. The ideal version of this function would look like this:

```python
class Ensemble:
    members: Sequence[EnsembleMember]

    def disagreement(self, input: Array) -> float:
        # TODO: Implement this!
        pass

disagreement = Ensemble.load("some/path/to/models").disagreement(input=np.array([1.0, 2.0, 3.0]))
```

Testing that is, however, somewhat tricky, since it's hard to know what exactly each member of the ensemble will predict. Moreover, we just want to make sure the disagreement computation is done correctly, and leave the correctness of the members' predictions a problem for another day. Fortunately, this is easy to do:

1. Abstract away a member of the ensemble as a "black box":

```python
class EnsembleMember(Protocol):
    def predict(self, input: Array) -> Array:
        """Predict the number of duplicate outfits an IT person has in their wardrobe, given their git commit history."""
        ...
```

2. Create a stub implementation for the test that is easy to control:

```python
@dataclass(frozen=True)
class StubEnsembleMember:
    prediction: Array

    @staticmethod
    def predicting(prediction: Array) -> "StubEnsembleMember":
        return StubEnsembleMember(prediction=prediction)

    def predict(self, input: Array) -> Array:
        return self.prediction
```

3. Test the disagreement computation using the stub:

```python
def test_that_ensemble_disagreement_is_zero_when_all_members_predict_the_same() -> None:
    ensemble = Ensemble(members=[
        StubEnsembleMember.predicting(prediction=np.array([1.0, 2.0, 3.0])),
        StubEnsembleMember.predicting(prediction=np.array([1.0, 2.0, 3.0])),
        StubEnsembleMember.predicting(prediction=np.array([1.0, 2.0, 3.0])),
    ])

    disagreement = ensemble.disagreement(input= np.array([1.0, 2.0, 3.0]))

    assert disagreement == approx(0.0)
```

In this case the stub feels like a mock, but it's really just a very simple fake implementation of an ensemble member. In other cases, you may want to use a more complex fake implementation that has some "toy" logic in it, but is still much easier to reason about than the real thing. Regardless, the test is now fully in your control.

### Docstrings

Use Google-style docstrings:

```python
def simulate(
    self,
    *,
    inputs: ControlInputBatch,
    initial_state: State,
) -> StateBatch:
    """Simulate the dynamical model forward in time.

    Args:
        inputs: Control inputs for each rollout.
        initial_state: Starting state for all rollouts.

    Returns:
        State trajectories for each rollout.

    Example:
        >>> model = bicycle.dynamical(time_step_size=0.1, wheelbase=2.5)
        >>> states = model.simulate(inputs=samples, initial_state=start)
    """
```

Try to avoid documenting things that you think are obvious, or in a way that does not add new information.

```python
# Good
class InputBatch:
    """Batch of control input sequences for multiple rollouts."""
    
    ...

    @property
    def horizon(self) -> int:
        """The number of time steps in a single control input sequence of this batch."""
        ...  

# Bad
class InputBatch:
    """An instance of the InputBatch class."""

    ...

    @property
    def horizon(self) -> int:
        """Returns the horizon of this InputBatch."""
        ...
```
