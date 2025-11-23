# VisProbe: Comprehensive API Reference

This document provides a complete reference for the VisProbe API, including all decorators, strategies, properties, and usage patterns.

## Table of Contents

1. [Core Decorators](#core-decorators)
2. [Perturbation Strategies](#perturbation-strategies)
3. [Robustness Properties](#robustness-properties)
4. [Data Handling](#data-handling)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)

## Core Decorators

### @given

Defines a test with fixed perturbation parameters.

```python
@given(
    strategy: Strategy,
    vectorized: bool = False,
    noise_sweep: Optional[Dict[str, Any]] = None,
    resolutions: Optional[List[Tuple[int, int]]] = None,
    top_k: Optional[int] = 5,
    property_name: Optional[str] = None
)
```

**Parameters:**
- `strategy`: The perturbation strategy to apply
- `vectorized`: Whether to process samples in batches (default: False)
- `noise_sweep`: Configuration for noise sensitivity analysis
- `resolutions`: List of (height, width) tuples for resolution analysis
- `top_k`: Number of top predictions to analyze (default: 5)
- `property_name`: Optional display name for the property

**Example:**
```python
@given(strategy=FGSMStrategy(eps=0.03))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_adversarial_robustness(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)
```

### @search

Defines an adaptive search test to find failure thresholds.

```python
@search(
    strategy: Callable[[float], Strategy] | Strategy,
    initial_level: float,
    step: float,
    min_step: float = 1e-5,
    max_queries: int = 500,
    resolutions: Optional[List[Tuple[int, int]]] = None,
    noise_sweep: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = 5,
    reduce: Optional[str] = 'all',
    property_name: Optional[str] = None
)
```

**Parameters:**
- `strategy`: Strategy factory function or fixed strategy
- `initial_level`: Starting perturbation level
- `step`: Initial step size for search
- `min_step`: Minimum step size (search stops when reached)
- `max_queries`: Maximum model queries allowed
- `reduce`: How to aggregate results ('all', 'any', 'mean')

**Example:**
```python
@search(
    strategy=lambda level: FGSMStrategy(eps=level),
    initial_level=0.01,
    step=0.005,
    max_queries=100
)
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def find_failure_threshold(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

### @model

Attaches a model to a test function.

```python
@model(
    model_obj: Any,
    capture_intermediate_layers: Optional[List[str]] = None
)
```

**Parameters:**
- `model_obj`: The PyTorch model to test
- `capture_intermediate_layers`: List of layer names to capture for analysis

### @data_source

Provides data configuration for a test.

```python
@data_source(
    data_obj: Any,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    class_names: Optional[List[str]] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
)
```

**Parameters:**
- `data_obj`: Dataset or list of samples
- `collate_fn`: Function to batch samples
- `class_names`: List of class names for visualization
- `mean`: Normalization mean (defaults to ImageNet)
- `std`: Normalization standard deviation (defaults to ImageNet)

## Perturbation Strategies

### Adversarial Strategies

#### FGSMStrategy
Fast Gradient Sign Method attack.

```python
FGSMStrategy(
    eps: float = 2/255,
    targeted: bool = False,
    art_attack_kwargs: dict = None
)
```

#### PGDStrategy
Projected Gradient Descent attack.

```python
PGDStrategy(
    eps: float,
    eps_step: float | None = None,
    max_iter: int = 100,
    **kwargs
)
```

#### BIMStrategy
Basic Iterative Method (iterated FGSM).

```python
BIMStrategy(
    eps: float,
    eps_step: float | None = None,
    max_iter: int = 10,
    **kwargs
)
```

#### APGDStrategy
Auto-Projected Gradient Descent.

```python
APGDStrategy(
    eps: float,
    max_iter: int = 100,
    **kwargs
)
```

#### SquareAttackStrategy
Score-based Square Attack.

```python
SquareAttackStrategy(
    eps: float,
    max_iter: int = 5000,
    **kwargs
)
```

### Natural Perturbation Strategies

#### GaussianNoiseStrategy
Additive Gaussian noise.

```python
GaussianNoiseStrategy(
    std_dev: float,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    seed: Optional[int] = None
)
```

#### BrightnessStrategy
Brightness adjustment.

```python
BrightnessStrategy(brightness_factor: float)
```

#### RotateStrategy
Image rotation.

```python
RotateStrategy(angle: float)
```

## Robustness Properties

### LabelConstant
Ensures the top-1 prediction remains unchanged.

```python
# Usage in test function
assert LabelConstant.evaluate(original, perturbed)

# Or create instance
prop = LabelConstant()
assert prop(original, perturbed)
```

### TopKStability
Checks stability of top-k predictions with multiple modes.

```python
TopKStability(
    k: int = 5,
    mode: Literal["overlap", "containment", "jaccard"] = "overlap",
    min_overlap: int = 3,
    require_containment: bool = True,
    min_jaccard: float = 0.4
)
```

**Modes:**
- `overlap`: Require at least `min_overlap` common classes in top-k sets
- `containment`: Original top-1 must be in perturbed top-k
- `jaccard`: Jaccard index between sets must be ≥ `min_jaccard`

### ConfidenceDrop
Limits confidence decrease in top prediction.

```python
ConfidenceDrop(max_drop: float = 0.3)
```

### L2Distance
Constrains L2 distance between output logits.

```python
L2Distance(max_delta: float = 1.0)
```

## Data Handling

### Built-in Utilities

```python
from visprobe.api.utils import cifar10_data_source

# Create CIFAR-10 data source
data_obj, collate_fn, class_names, mean, std = cifar10_data_source(
    dataset_or_subset,
    normalized=True,
    meta_path='./data/cifar-10-batches-py/batches.meta'
)
```

### Custom Data Sources

```python
# For custom datasets
my_data = [img1, img2, img3, ...]  # List of tensors

def my_collate_fn(batch_list):
    return torch.stack(batch_list, dim=0)

@data_source(
    data_obj=my_data,
    collate_fn=my_collate_fn,
    class_names=['class1', 'class2', ...],
    mean=[0.485, 0.456, 0.406],  # Custom normalization
    std=[0.229, 0.224, 0.225]
)
```

## Configuration

### Device Management

For automatic device configuration:

```python
import visprobe.auto_init  # Add at top of test files
```

Or manual configuration:

```python
import os
os.environ["VISPROBE_DEVICE"] = "cpu"  # or "cuda", "mps"
os.environ["VISPROBE_PREFER_GPU"] = "1"  # Enable GPU preference
```

### Environment Variables

- `VISPROBE_DEVICE`: Force specific device (cpu, cuda, mps)
- `VISPROBE_PREFER_GPU`: Enable GPU preference if available
- `VF_THREADS`: Number of PyTorch threads
- `OMP_NUM_THREADS`: OpenMP threads
- `VISPROBE_DEBUG`: Enable debug output

## Advanced Usage

### Multiple Properties

Test multiple properties simultaneously:

```python
@given(strategy=FGSMStrategy(eps=0.03))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_comprehensive_robustness(original, perturbed):
    # All must pass
    assert LabelConstant.evaluate(original, perturbed)
    assert ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3)
    assert TopKStability.evaluate(original, perturbed, k=5, min_overlap=3)
```

### Composite Strategies

Apply multiple perturbations sequentially:

```python
composite_strategy = [
    GaussianNoiseStrategy(std_dev=0.01),
    BrightnessStrategy(brightness_factor=1.1)
]

@given(strategy=composite_strategy)
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_multiple_perturbations(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)
```

### Custom Properties

Create custom robustness properties:

```python
from visprobe.properties.base import Property

class CustomProperty(Property):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, original: Any, perturbed: Any) -> bool:
        # Your custom logic here
        return custom_check(original, perturbed, self.threshold)

    def __str__(self) -> str:
        return f"CustomProperty(threshold={self.threshold})"

# Usage
@given(strategy=GaussianNoiseStrategy(std_dev=0.05))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_custom_property(original, perturbed):
    assert CustomProperty.evaluate(original, perturbed, threshold=0.7)
```

### Intermediate Layer Analysis

Capture and analyze intermediate layer activations:

```python
@given(strategy=FGSMStrategy(eps=0.03))
@model(my_model, capture_intermediate_layers=['layer1', 'layer2', 'layer3'])
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_with_layer_analysis(original, perturbed):
    # original["features"] and perturbed["features"] contain layer activations
    assert LabelConstant.evaluate(original, perturbed)
```

## CLI Usage

### Running Tests

```bash
# Run tests and save results
visprobe run test_my_model.py

# Run with specific device
visprobe run test_my_model.py --device cuda

# Keep previous results
visprobe run test_my_model.py --keep
```

### Visualization

```bash
# Launch interactive dashboard
visprobe visualize test_my_model.py

# Automatically runs test if no results found
visprobe visualize test_my_model.py --device cpu
```

## Error Handling

### Common Issues

1. **Device Mismatch**: Use `import visprobe.auto_init` or set `VISPROBE_DEVICE=cpu`
2. **Memory Issues**: Reduce batch size or use `vectorized=False`
3. **Missing Dependencies**: Install with `pip install adversarial-robustness-toolbox`

### Debugging

```python
import os
os.environ["VISPROBE_DEBUG"] = "1"  # Enable debug output
```

## Best Practices

1. **Always use `import visprobe.auto_init`** for stable device management
2. **Start with small datasets** for testing
3. **Use appropriate perturbation levels** for your domain
4. **Test multiple properties** for comprehensive evaluation
5. **Document your test intentions** with clear function names and comments
6. **Use version control** for reproducible experiments

## Integration Examples

### Research Pipeline

```python
import visprobe.auto_init
from visprobe import given, model, data_source
from visprobe.strategies import FGSMStrategy, PGDStrategy
from visprobe.properties import LabelConstant, ConfidenceDrop

# Load your model and data
my_model = load_pretrained_model()
my_data = load_test_dataset()

@given(strategy=FGSMStrategy(eps=0.031))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=default_collate)
def test_fgsm_robustness(original, perturbed):
    """Test robustness against FGSM attacks."""
    return (LabelConstant.evaluate(original, perturbed) and
            ConfidenceDrop.evaluate(original, perturbed, max_drop=0.5))

@given(strategy=PGDStrategy(eps=0.031, eps_step=0.007, max_iter=40))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=default_collate)
def test_pgd_robustness(original, perturbed):
    """Test robustness against stronger PGD attacks."""
    return LabelConstant.evaluate(original, perturbed)

if __name__ == "__main__":
    # Run all tests
    test_fgsm_robustness()
    test_pgd_robustness()
```

### Production Monitoring

```python
import visprobe.auto_init
from visprobe import given, model, data_source
from visprobe.strategies import GaussianNoiseStrategy
from visprobe.properties import ConfidenceDrop, TopKStability

@given(strategy=GaussianNoiseStrategy(std_dev=0.02))
@model(production_model)
@data_source(data_obj=validation_samples, collate_fn=production_collate)
def monitor_noise_robustness(original, perturbed):
    """Monitor model robustness to natural noise in production."""
    return (ConfidenceDrop.evaluate(original, perturbed, max_drop=0.2) and
            TopKStability.evaluate(original, perturbed, k=3, min_overlap=2))

# Integrate into monitoring pipeline
def daily_robustness_check():
    report = monitor_noise_robustness()
    if report.robust_accuracy < 0.95:  # Alert threshold
        send_alert(f"Robustness dropped to {report.robust_accuracy:.2%}")
    return report
```

This comprehensive reference covers all major aspects of the VisProbe API. For additional examples and use cases, see the test files in the repository.

