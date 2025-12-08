# Design Rationale

This document explains the "why" behind VisProbe's design decisions.

## Core Philosophy

### 1. Declarative Over Imperative

**Decision**: Use decorators for test configuration rather than explicit class instantiation.

**Why**:
```python
# VisProbe (declarative)
@model(my_model)
@data_source(data)
@search(strategy=...)
def test_robustness(original, perturbed):
    return property_holds(original, perturbed)

# Alternative (imperative)
runner = TestRunner(model=my_model, data=data)
runner.set_strategy(...)
runner.run_search(property_fn=lambda o, p: property_holds(o, p))
```

**Benefits**:
- **Readability**: Test intent is immediately clear
- **Composability**: Decorators can be mixed and matched
- **Familiarity**: Python developers know decorators
- **Testability**: Functions remain pure and testable

**Tradeoffs**:
- Less discoverable for IDE autocomplete (mitigated by documentation)
- Magic happening behind the scenes (mitigated by clear separation)

### 2. Properties as Callables, Not Assertions

**Decision**: Properties return `bool` instead of raising `AssertionError`.

**Why**:
```python
# VisProbe
class LabelConstant(Property):
    def __call__(self, original, perturbed):
        return original_label == perturbed_label

# Alternative
def label_constant(original, perturbed):
    assert original_label == perturbed_label
```

**Benefits**:
- **Search-friendly**: Can easily aggregate pass/fail counts
- **Composable**: `property1() and property2()`
- **Framework agnostic**: Works with any testing framework
- **Informative**: Can return partial scores, not just pass/fail

**Backward Compatibility**: We still support assertion-based tests for compatibility with existing codebases.

### 3. Strategy Pattern for Perturbations

**Decision**: Strategies are objects with a `generate()` method.

**Why**:
```python
class Strategy:
    def generate(self, imgs, model, level=None):
        raise NotImplementedError
```

**Benefits**:
- **Open-closed principle**: Easy to add new strategies without modifying core code
- **Testable**: Each strategy can be unit tested independently
- **Composable**: Strategies can be chained via `CompositeStrategy`
- **Stateful**: Strategies can cache expensive computations (e.g., ART classifiers)

**Why not functions?**
- Functions can't cache state efficiently
- Hard to compose complex perturbation pipelines
- Difficult to introspect (e.g., what parameters were used?)

### 4. CPU as Default Device

**Decision**: Default to CPU, require explicit opt-in for GPU.

**Why**:
```python
def get_default_device():
    if os.environ.get("VISPROBE_PREFER_GPU") == "1":
        # Try CUDA, ROCm, MPS...
    return torch.device("cpu")  # Safe default
```

**Benefits**:
- **Maximum compatibility**: Works everywhere without configuration
- **Stability**: Avoids GPU OOM on large models
- **Reproducibility**: CPU results are more deterministic
- **CI/CD friendly**: GitHub Actions, GitLab CI don't have GPUs

**When to use GPU**: Large models, adversarial attacks, real-time constraints

### 5. Lazy Imports for Optional Dependencies

**Decision**: Import ART only when adversarial strategies are used.

**Why**:
```python
try:
    from art.attacks.evasion import FastGradientMethod
    _ART_AVAILABLE = True
except ImportError:
    _ART_AVAILABLE = False
```

**Benefits**:
- **Reduced dependencies**: Users don't need ART for natural transformations
- **Faster imports**: Only pay for what you use
- **Easier installation**: `pip install visprobe` works without ART

**Clear error messages**:
```python
if not _ART_AVAILABLE:
    raise ImportError(
        "Adversarial strategies require ART.\n"
        "Install with: pip install adversarial-robustness-toolbox"
    )
```

## Search Algorithm Design

### 1. Why Four Search Modes?

**Adaptive**: Step-halving for unknown failure regions
- **Use case**: Exploratory testing, unknown perturbation space
- **Complexity**: O(log(1/ε)) where ε is threshold precision
- **Queries**: ~20-50 for typical ranges

**Binary**: True binary search for known bounds
- **Use case**: Known failure bounds, precise threshold needed
- **Complexity**: O(log((hi-lo)/min_step))
- **Queries**: ~10-15 for typical ranges

**Grid**: Uniform sampling
- **Use case**: Visualization, creating robustness curves
- **Complexity**: O(n) where n is number of levels
- **Queries**: Exactly `num_levels`

**Random**: Stochastic sampling
- **Use case**: High-dimensional spaces, stochastic strategies
- **Complexity**: O(n) where n is number of samples
- **Queries**: Exactly `num_samples`

**Why not just one?**
- Different use cases have different needs
- Binary is fastest for known bounds
- Grid is best for visualization
- Adaptive is best for exploration

### 2. Per-Sample Bracketing

**Decision**: Track `[last_pass, first_fail]` for each sample independently.

**Why**:
```python
last_pass_levels = [None] * batch_size
first_fail_levels = [None] * batch_size

# Update per sample
for i, passed in enumerate(passed_mask):
    if passed:
        last_pass_levels[i] = level
    elif first_fail_levels[i] is None:
        first_fail_levels[i] = level
```

**Benefits**:
- **Tighter bounds**: Midpoint estimate `(last_pass + first_fail) / 2`
- **Per-sample thresholds**: See distribution across batch
- **Robustness metrics**: Compute quantiles (p05, median, p95)

**Tradeoff**: Slightly more memory (negligible for typical batch sizes)

### 3. Reducer Specifications

**Decision**: Support `"all"`, `"any"`, `"frac>=X"` for batch aggregation.

**Why**:
```python
reduce="all"       # All samples must pass (strict)
reduce="any"       # At least one passes (lenient)
reduce="frac>=0.8" # 80% must pass (flexible)
```

**Benefits**:
- **Flexibility**: Different robustness criteria
- **Meaningful thresholds**: "80% robust at level X" is actionable
- **Statistical testing**: Easy to compute confidence intervals

**Use cases**:
- `"all"`: Worst-case robustness
- `"frac>=0.9"`: Percentile robustness
- `"any"`: Best-case / existence proofs

## Property Design

### 1. Why Property Classes Instead of Functions?

**Decision**: Properties are classes inheriting from `Property`.

**Why**:
```python
class TopKStability(Property):
    def __init__(self, k=5, min_overlap=3):
        self.k = k
        self.min_overlap = min_overlap

    def __call__(self, original, perturbed):
        # Use self.k, self.min_overlap
```

**Benefits**:
- **Parameterization**: Each property can have configuration
- **Reusability**: Same property with different parameters
- **Introspection**: Can inspect property configuration
- `__str__` for readable reports

**Alternative considered**: Functions with closures
```python
def make_topk(k, min_overlap):
    def property_fn(original, perturbed):
        # ...
    return property_fn
```

**Why classes won**: Better introspection, clearer intent, easier to extend

### 2. Why `evaluate()` Classmethod?

**Decision**: Provide `Property.evaluate(original, perturbed, **kwargs)` convenience method.

**Why**:
```python
# Concise for one-off use
LabelConstant.evaluate(original, perturbed)

# Equivalent to
prop = LabelConstant()
prop(original, perturbed)
```

**Benefits**:
- **Convenience**: Don't need to instantiate for parameterless properties
- **Inline parameters**: `TopKStability.evaluate(original, perturbed, k=5)`
- **Familiar pattern**: Similar to `staticmethod` / `classmethod`

## Report Design

### 1. Why Dataclasses?

**Decision**: Use `@dataclass` for `Report`, `ImageData`, etc.

**Why**:
```python
@dataclass
class Report:
    test_name: str
    test_type: str
    runtime: float
    # ... many fields
```

**Benefits**:
- **Type safety**: Static type checking with mypy
- **Auto methods**: `__init__`, `__repr__`, `__eq__` generated
- **Easy serialization**: `asdict()` for JSON export
- **Documentation**: Field types serve as documentation

**Alternative considered**: Plain dicts
```python
report = {
    "test_name": "...",
    "test_type": "...",
    # Easy to mistype keys!
}
```

**Why dataclasses won**: Type safety catches bugs at development time

### 2. Why Separate Images from JSON?

**Decision**: Save images as PNG files, reference by path in JSON.

**Why**:
```python
# Not this (huge JSON files)
{
    "original_image": {
        "image_b64": "iVBORw0KGgoAAAANSUhEUg..."  # 100KB+
    }
}

# This (small JSON, separate images)
{
    "original_image": {
        "image_path": "/tmp/visprobe_results/test.original.png",
        "prediction": "cat",
        "confidence": 0.95
    }
}
```

**Benefits**:
- **Small JSON files**: Easy to parse, version control
- **Standard formats**: PNGs can be viewed anywhere
- **Efficient**: No base64 encoding/decoding overhead
- **Cacheable**: Images can be cached separately

**Tradeoff**: Two files instead of one (acceptable for local filesystem)

### 3. Why Both JSON and CSV?

**Decision**: Save both `test.json` (full report) and `test.csv` (per-sample metrics).

**Why**:

**JSON**: Complete information
```json
{
    "test_name": "...",
    "failure_threshold": 0.05,
    "search_path": [...],
    "per_sample": [...]
}
```

**CSV**: Easy plotting
```csv
index,passed,threshold_estimate,confidence_drop,...
0,True,0.048,0.05,...
1,False,0.032,0.12,...
```

**Benefits**:
- **JSON**: Machine-readable, complete
- **CSV**: Excel/Pandas/R friendly
- **Both**: Use the right tool for the job

## Threading and Concurrency

### 1. Thread-Local Test Registry

**Decision**: Use `threading.local()` for test registration.

**Why**:
```python
class TestRegistry:
    _local = threading.local()

    @classmethod
    def get_given_tests(cls):
        if not hasattr(cls._local, "given_tests"):
            cls._local.given_tests = []
        return cls._local.given_tests
```

**Benefits**:
- **Thread-safe**: Each thread has its own test list
- **No locks**: No contention, no deadlocks
- **Parallel testing**: Can run tests in parallel safely

**Why not global list?**
```python
GIVEN_TESTS = []  # Race conditions in parallel execution!
```

### 2. PyTorch Hooks for Query Counting

**Decision**: Use `register_forward_hook()` instead of monkey-patching.

**Why**:
```python
# VisProbe
class QueryCounter:
    def __enter__(self):
        self._hook = self.model.register_forward_hook(self._counter)

# Alternative (dangerous)
original_forward = model.forward
model.forward = lambda x: (counter.inc(), original_forward(x))
```

**Benefits**:
- **Thread-safe**: Hooks are instance-specific
- **No pollution**: Doesn't modify global state
- **Automatic cleanup**: Hooks removed in `__exit__`
- **Composable**: Multiple hooks can coexist

**Tradeoff**: Slightly more verbose (worth it for safety)

## Performance Optimizations

### 1. Tensor Caching

**Decision**: Cache normalized mean/std tensors per device.

**Why**:
```python
# Without caching (slow)
def generate(self, imgs, model):
    mean = torch.tensor(self.mean, device=imgs.device).view(1, 3, 1, 1)
    # Recreated every call!

# With caching (fast)
def _get_stats(self, device):
    if device not in self._stats_cache:
        self._stats_cache[device] = (mean_tensor, std_tensor)
    return self._stats_cache[device]
```

**Benefits**:
- **Faster**: No repeated tensor allocation
- **Less memory**: Shared across calls
- **Device-aware**: Separate cache per device

**When it matters**: Strategies called hundreds of times in search

### 2. Batch Property Evaluation

**Decision**: Support both per-sample and vectorized property evaluation.

**Why**:
```python
# Per-sample (flexible, slower)
for i in range(batch_size):
    passed = property_fn(
        {"output": clean[i:i+1]},
        {"output": pert[i:i+1]}
    )

# Vectorized (faster)
passed = property_fn(
    {"output": clean},  # Full batch
    {"output": pert}
)
```

**Benefits**:
- **Vectorized**: 10-100x faster for simple properties
- **Flexible**: Per-sample works for complex logic

**Use cases**:
- Vectorized: `LabelConstant` on large batches
- Per-sample: Complex properties with conditionals

### 3. Detached Feature Caching

**Decision**: Detach intermediate layer outputs immediately.

**Why**:
```python
def hook(module, input, output):
    self._features[name] = output.detach()  # Detach!
```

**Benefits**:
- **Memory**: Doesn't keep computation graph in memory
- **Speed**: No gradient tracking overhead
- **Safety**: Can't accidentally backprop through stored features

**When it matters**: Models with many intermediate layers

## Error Handling Philosophy

### 1. Fail Fast on Configuration

**Decision**: Validate all parameters at decoration time, not runtime.

**Why**:
```python
def search(..., min_step=1e-5, step=0.002):
    if min_step > step:
        raise ValidationError("min_step must be <= step")
    # Fails immediately when @search is applied
```

**Benefits**:
- **Immediate feedback**: See errors when writing code
- **Better error messages**: Clear parameter names, not runtime state
- **IDE support**: Type checkers catch errors

**Alternative**: Validate at `run()` time
- Bad UX: Error after test already started
- Confusing: Error context is lost

### 2. Graceful Degradation for Analysis

**Decision**: Analysis features return `None` on failure, don't crash the test.

**Why**:
```python
try:
    corruption_results = run_corruption_sweep(...)
except Exception:
    corruption_results = None  # Don't fail entire test
```

**Benefits**:
- **Robustness**: One analysis failure doesn't break everything
- **Useful reports**: Get partial results even if something fails
- **Debugging**: Can inspect what succeeded

**When to fail hard**: Core functionality (model forward, property eval)

### 3. Informative Import Errors

**Decision**: Provide actionable error messages for missing dependencies.

**Why**:
```python
raise ImportError(
    "Adversarial strategies require ART.\n"
    "Install with: pip install adversarial-robustness-toolbox\n"
    f"Original error: {e}"
)
```

**Benefits**:
- **Actionable**: User knows exactly what to do
- **Context**: Original error preserved for debugging
- **Helpful**: Don't just say "module not found"

## Future-Proofing

### 1. Why Not Use PyTest Directly?

**Decision**: Custom decorator API instead of PyTest integration.

**Why**:
- **Framework agnostic**: Works with unittest, nose, pytest, etc.
- **Specialized**: Property-based testing needs different primitives
- **Interactive**: Dashboard visualization isn't test-framework specific
- **Simpler**: Users don't need to learn pytest fixtures

**Can still use with pytest**:
```python
def test_with_pytest():
    report = my_visprobe_test()
    assert report.robust_accuracy > 0.9
```

### 2. Why Not Use Hypothesis?

**Decision**: Custom search algorithms instead of Hypothesis strategies.

**Why**:
- **Specialized**: Hypothesis is for property-based testing of code, not ML models
- **Query budgets**: Need explicit control over number of model calls
- **Threshold search**: Hypothesis doesn't do adaptive threshold finding
- **Domain-specific**: Image perturbations need special handling (normalization, etc.)

**Hypothesis is great for**: Testing VisProbe itself!

### 3. Extensibility Points

**Design decision**: Make it easy to extend without forking.

**Extension points**:
1. **Custom strategies**: Inherit from `Strategy`
2. **Custom properties**: Inherit from `Property`
3. **Custom search modes**: Add function to `search_modes.py`
4. **Custom analysis**: Add function to `analysis_utils.py`

**Future plugin system** (planned):
```python
# Register third-party strategies
@visprobe.register_strategy("my_custom_strategy")
class MyStrategy(Strategy):
    ...
```

## Lessons Learned

### 1. Start Simple, Add Complexity

**Initial version**: Only `@given` decorator, one search mode
**Current version**: `@given` and `@search`, four search modes

**Lesson**: Ship MVP, iterate based on user feedback

### 2. Optimize After Profiling

**Early concern**: "Hooks might be slow"
**Reality**: <1% overhead for typical models

**Lesson**: Profile first, optimize second

### 3. Documentation is Design

**Observation**: Hard-to-document APIs are hard-to-use APIs

**Lesson**: If you can't explain it clearly, redesign it

### 4. Examples > Explanations

**What works**: Complete, runnable code examples
**What doesn't**: Abstract descriptions of capabilities

**Lesson**: Show, don't tell

## Alternatives Considered

### 1. Configuration Files vs Decorators

**Considered**:
```yaml
# config.yaml
tests:
  - name: test_noise
    model: resnet18
    strategy: gaussian_noise
    level: 0.05
```

**Why decorators won**: Python is more expressive, type-safe, composable

### 2. Callbacks vs Hooks

**Considered**: User-provided callbacks for search events
```python
@search(
    strategy=...,
    on_pass=lambda level: print(f"Passed at {level}"),
    on_fail=lambda level: print(f"Failed at {level}")
)
```

**Why not**: Adds complexity, unclear benefit over inspection of final report

### 3. Async API

**Considered**: `async def` for long-running searches
```python
@search(...)
async def test_robustness(original, perturbed):
    return await property_async(original, perturbed)
```

**Why not**: Most models are synchronous, adds complexity for minimal benefit

Could revisit for distributed/remote model inference.

## Conclusion

VisProbe's design prioritizes:
1. **User experience**: Clean API, clear errors, good docs
2. **Correctness**: Type safety, validation, reproducibility
3. **Performance**: Smart defaults, optional optimizations
4. **Extensibility**: Easy to add strategies/properties/search modes

The goal is to make robustness testing as easy as writing unit tests.
