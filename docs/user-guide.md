# VisProbe User Guide

This guide will walk you through using VisProbe to test the robustness of your machine learning models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Writing Tests](#writing-tests)
5. [Perturbation Strategies](#perturbation-strategies)
6. [Robustness Properties](#robustness-properties)
7. [Search Modes](#search-modes)
8. [Visualization](#visualization)
9. [Advanced Usage](#advanced-usage)
10. [Best Practices](#best-practices)

## Installation

### Basic Installation

```bash
pip install visprobe
```

### With Optional Dependencies

```bash
# For adversarial attacks
pip install visprobe[adversarial]

# For visualization
pip install visprobe[viz]

# Everything
pip install visprobe[all]
```

### From Source

```bash
git clone https://github.com/your-repo/visprobe.git
cd visprobe
pip install -e .
```

## Quick Start

Here's a minimal example to test a model's robustness to Gaussian noise:

```python
import torch
from torchvision import models
from visprobe import model, data_source, given
from visprobe.strategies.image import GaussianNoiseStrategy
from visprobe.properties.classification import LabelConstant

# Load a pretrained model
resnet = models.resnet18(pretrained=True)
resnet.eval()

# Prepare test data (4D tensor: batch x channels x height x width)
test_images = torch.randn(8, 3, 224, 224)

@model(resnet)
@data_source(test_images, class_names=["class_0", "class_1", ...])
@given(strategy=GaussianNoiseStrategy(std_dev=0.05))
def test_noise_robustness(original, perturbed):
    """Test if model predictions remain constant under small noise."""
    return LabelConstant.evaluate(original, perturbed)

# Run the test
if __name__ == "__main__":
    report = test_noise_robustness()
    print(f"Robust accuracy: {report.robust_accuracy:.2%}")
```

## Core Concepts

### 1. Decorators

VisProbe uses Python decorators to configure tests:

- `@model()` - Attaches the model to test
- `@data_source()` - Provides test data
- `@given()` - Fixed perturbation test
- `@search()` - Adaptive threshold search

### 2. Test Functions

Your test function receives two dictionaries:

```python
def test_robustness(original, perturbed):
    # original = {"output": clean_logits}
    # perturbed = {"output": perturbed_logits}
    return property_holds(original, perturbed)
```

### 3. Strategies

Strategies define *how* to perturb inputs. Examples:
- `GaussianNoiseStrategy` - Add random noise
- `FGSMStrategy` - Fast Gradient Sign Method attack
- `BrightnessStrategy` - Adjust image brightness

### 4. Properties

Properties define *what* robustness means for your task:
- `LabelConstant` - Top-1 prediction stays the same
- `TopKStability` - Top-k predictions overlap
- `ConfidenceDrop` - Confidence doesn't drop too much

## Writing Tests

### Fixed Perturbation Tests (@given)

Use `@given` when you want to test robustness at a specific perturbation level:

```python
@model(my_model)
@data_source(test_data)
@given(strategy=GaussianNoiseStrategy(std_dev=0.1))
def test_fixed_noise(original, perturbed):
    """Test if model is robust to σ=0.1 Gaussian noise."""
    return LabelConstant.evaluate(original, perturbed)
```

### Threshold Search Tests (@search)

Use `@search` to automatically find the failure threshold:

```python
@model(my_model)
@data_source(test_data)
@search(
    strategy=lambda level: GaussianNoiseStrategy(std_dev=level),
    initial_level=0.001,
    step=0.002,
    min_step=1e-5,
    max_queries=500
)
def test_noise_threshold(original, perturbed):
    """Find the noise level where model starts failing."""
    return LabelConstant.evaluate(original, perturbed)
```

**Key Parameters:**
- `strategy` - Can be a Strategy instance or a factory function `lambda level: Strategy(level)`
- `initial_level` - Starting perturbation level
- `step` - Step size for adaptive search
- `min_step` - Minimum step size before stopping
- `max_queries` - Maximum model queries allowed

## Perturbation Strategies

### Natural Transformations

#### Gaussian Noise

```python
from visprobe.strategies.image import GaussianNoiseStrategy

# Fixed noise level
strategy = GaussianNoiseStrategy(std_dev=0.05)

# For search (factory function)
strategy = lambda level: GaussianNoiseStrategy(std_dev=level)
```

#### Brightness Adjustment

```python
from visprobe.strategies.image import BrightnessStrategy

# factor > 1.0 brightens, < 1.0 darkens
strategy = BrightnessStrategy(brightness_factor=1.2)
```

#### Contrast Adjustment

```python
from visprobe.strategies.image import ContrastStrategy

# factor > 1.0 increases contrast, < 1.0 decreases
strategy = ContrastStrategy(contrast_factor=0.8)
```

#### Rotation

```python
from visprobe.strategies.image import RotateStrategy

# Angle in degrees (counter-clockwise)
strategy = RotateStrategy(angle=15.0)
```

### Adversarial Attacks

Requires `pip install adversarial-robustness-toolbox`

#### FGSM (Fast Gradient Sign Method)

```python
from visprobe.strategies.adversarial import FGSMStrategy

strategy = lambda eps: FGSMStrategy(eps=eps)
```

#### PGD (Projected Gradient Descent)

```python
from visprobe.strategies.adversarial import PGDStrategy

strategy = lambda eps: PGDStrategy(
    eps=eps,
    eps_step=eps/10,
    max_iter=100
)
```

#### Auto-PGD

```python
from visprobe.strategies.adversarial import APGDStrategy

# Adaptive step-size PGD
strategy = lambda eps: APGDStrategy(eps=eps, max_iter=100)
```

### Composite Strategies

Chain multiple perturbations:

```python
from visprobe.strategies.base import Strategy

# Apply noise then brightness
strategy = [
    GaussianNoiseStrategy(std_dev=0.02),
    BrightnessStrategy(brightness_factor=1.1)
]
```

### Custom Strategies

Create your own:

```python
from visprobe.strategies.base import Strategy
import torch

class SaltPepperNoise(Strategy):
    def __init__(self, density=0.05):
        self.density = density

    def generate(self, imgs, model, level=None):
        density = level if level is not None else self.density
        noise = torch.rand_like(imgs)
        imgs_noisy = imgs.clone()
        imgs_noisy[noise < density/2] = 0  # Pepper
        imgs_noisy[noise > 1 - density/2] = 1  # Salt
        return imgs_noisy
```

## Robustness Properties

### Classification Properties

#### LabelConstant

Top-1 prediction must remain the same:

```python
from visprobe.properties.classification import LabelConstant

def test_function(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

#### TopKStability

Top-k predictions must overlap:

```python
from visprobe.properties.classification import TopKStability

def test_function(original, perturbed):
    # At least 3 of top-5 must overlap
    return TopKStability.evaluate(
        original, perturbed,
        k=5,
        mode="overlap",
        min_overlap=3
    )
```

Modes:
- `"overlap"` - Require minimum overlap count
- `"containment"` - Original top-1 in perturbed top-k
- `"jaccard"` - Jaccard index >= threshold

#### ConfidenceDrop

Confidence shouldn't drop too much:

```python
from visprobe.properties.classification import ConfidenceDrop

def test_function(original, perturbed):
    # Max 30% confidence drop allowed
    return ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3)
```

#### L2Distance

Output logits L2 distance:

```python
from visprobe.properties.classification import L2Distance

def test_function(original, perturbed):
    return L2Distance.evaluate(original, perturbed, max_delta=1.0)
```

### Custom Properties

```python
from visprobe.properties.base import Property
import torch

class MarginProperty(Property):
    def __init__(self, min_margin=0.1):
        self.min_margin = min_margin

    def __call__(self, original, perturbed):
        # Extract logits
        orig_logits = original["output"]
        pert_logits = perturbed["output"]

        # Check if margin is maintained
        orig_probs = torch.softmax(orig_logits, dim=-1)
        pert_probs = torch.softmax(pert_logits, dim=-1)

        # Top-2 margin
        orig_top2, _ = torch.topk(orig_probs, 2, dim=-1)
        pert_top2, _ = torch.topk(pert_probs, 2, dim=-1)

        orig_margin = orig_top2[:, 0] - orig_top2[:, 1]
        pert_margin = pert_top2[:, 0] - pert_top2[:, 1]

        return torch.all(pert_margin >= self.min_margin).item()
```

## Search Modes

### Adaptive Search (default)

Step-halving search that adapts to the failure point:

```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='adaptive',
    initial_level=0.001,
    step=0.002,
    min_step=1e-5
)
```

**How it works:**
1. Start at `initial_level`
2. If passes: increase by `step`
3. If fails: decrease by `step`, then halve `step`
4. Stop when `step < min_step`

**Best for:** Unknown failure regions

### Binary Search

O(log n) binary search for precise thresholds:

```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='binary',
    level_lo=0.0,
    level_hi=0.5,
    min_step=1e-5
)
```

**How it works:**
1. Start with bounds [level_lo, level_hi]
2. Test midpoint
3. If passes: search upper half
4. If fails: search lower half
5. Stop when range < min_step

**Best for:** Known bounds, precise thresholds

### Grid Search

Test evenly-spaced levels:

```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='grid',
    level_lo=0.0,
    level_hi=0.5,
    num_levels=21
)
```

**Best for:** Visualization, exploration

### Random Search

Sample random levels:

```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='random',
    level_lo=0.0,
    level_hi=0.5,
    num_samples=64
)
```

**Best for:** Large spaces, stochastic strategies

## Visualization

### Running the Dashboard

```bash
# Run tests and visualize
visprobe visualize my_test.py

# Or separately
visprobe run my_test.py
visprobe visualize my_test.py
```

### Dashboard Features

1. **Image Comparison** - Side-by-side original vs perturbed
2. **Search Path** - Visualization of search trajectory
3. **Metrics** - Failure threshold, queries used, etc.
4. **Analysis Tabs**:
   - Resolution Impact
   - Noise Sensitivity
   - Corruption Robustness
   - Top-K Analysis
   - Ensemble Analysis (if intermediate layers captured)

### Programmatic Access

```python
report = test_function()

# Access report fields
print(f"Failure threshold: {report.failure_threshold}")
print(f"Queries used: {report.model_queries}")
print(f"Robust accuracy: {report.robust_accuracy}")

# Export to JSON
json_str = report.to_json()

# Reports auto-save to temp directory
# Default: /tmp/visprobe_results/
# Override: export VISPROBE_RESULTS_DIR=/path/to/results
```

## Advanced Usage

### Capturing Intermediate Layers

```python
@model(
    my_resnet,
    capture_intermediate_layers=["layer4", "avgpool"]
)
@data_source(test_data)
@given(strategy=GaussianNoiseStrategy(std_dev=0.05))
def test_with_features(original, perturbed):
    # Now original/perturbed have features
    # original = {"output": logits, "features": {"layer4": ..., "avgpool": ...}}
    return LabelConstant.evaluate(original, perturbed)
```

The report will include layer-wise cosine similarity analysis.

### Custom Normalization

If your data uses custom normalization:

```python
from visprobe.api.utils import cifar10_data_source

# For CIFAR-10
dataset, collate, classes, mean, std = cifar10_data_source(
    cifar_dataset,
    normalized=True  # If dataset includes Normalize transform
)

@data_source(
    dataset,
    collate_fn=collate,
    class_names=classes,
    mean=mean,
    std=std
)
```

Or manually:

```python
@data_source(
    my_data,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
```

### Batch Reduction

Control how batch results are aggregated:

```python
@search(
    strategy=...,
    reduce="all"  # All samples must pass (default)
)

@search(
    strategy=...,
    reduce="any"  # At least one sample passes
)

@search(
    strategy=...,
    reduce="frac>=0.8"  # 80% of samples must pass
)
```

### Vectorized Properties

For faster evaluation on batches:

```python
@given(
    strategy=GaussianNoiseStrategy(std_dev=0.05),
    vectorized=True  # Evaluate entire batch at once
)
def test_vectorized(original, perturbed):
    # Receives full batch tensors
    return torch.all(
        torch.argmax(original["output"], dim=1) ==
        torch.argmax(perturbed["output"], dim=1)
    ).item()
```

### Device Selection

```bash
# Use CPU (default)
visprobe run test.py --device cpu

# Use CUDA
visprobe run test.py --device cuda

# Use Apple Silicon GPU
visprobe run test.py --device mps

# Use AMD GPU
visprobe run test.py --device hip

# Or via environment
export VISPROBE_DEVICE=cuda
export VISPROBE_PREFER_GPU=1  # Auto-select best GPU
```

### Reproducibility

```bash
# Set random seed
export VISPROBE_SEED=42

# Or in code
os.environ["VISPROBE_SEED"] = "42"
```

## Best Practices

### 1. Start with @given Tests

Before searching for thresholds, verify your property works:

```python
# First: sanity check
@given(strategy=NoOpStrategy())  # No perturbation
def test_sanity(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
# Should pass 100%

# Then: test at fixed level
@given(strategy=GaussianNoiseStrategy(std_dev=0.1))
def test_fixed(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

### 2. Use Binary Search When Possible

Binary search is much faster than adaptive for known bounds:

```python
# Slower: ~50 queries
@search(mode='adaptive', initial_level=0.001, step=0.002)

# Faster: ~10 queries
@search(mode='binary', level_lo=0.0, level_hi=0.1)
```

### 3. Set Realistic Query Budgets

```python
@search(
    strategy=...,
    max_queries=500  # Typical
)

# For expensive models
@search(
    strategy=...,
    max_queries=100
)
```

### 4. Use Property Factories

For reusable properties with different parameters:

```python
def make_topk_test(k, min_overlap):
    def test(original, perturbed):
        return TopKStability.evaluate(
            original, perturbed,
            k=k,
            mode="overlap",
            min_overlap=min_overlap
        )
    return test

@search(strategy=...)
def test_top5(original, perturbed):
    return make_topk_test(5, 3)(original, perturbed)
```

### 5. Organize Tests by Property

```python
# test_robustness/
#   test_label_constant.py
#   test_topk_stability.py
#   test_confidence.py
```

### 6. Document Your Tests

```python
@search(...)
def test_gaussian_noise_robustness(original, perturbed):
    """
    Tests robustness to additive Gaussian noise.

    Property: Top-1 label must remain constant
    Strategy: Gaussian noise with adaptive std search
    Expected: Threshold around 0.05-0.15 for ResNet-18
    """
    return LabelConstant.evaluate(original, perturbed)
```

### 7. Version Control Reports

```bash
# Add to .gitignore
/tmp/visprobe_results/

# Or commit for reproducibility
git add reports/
```

### 8. Use Meaningful Test Names

```python
# Good
@search(...)
def test_fgsm_l2_epsilon_threshold(original, perturbed):
    ...

# Bad
@search(...)
def test1(original, perturbed):
    ...
```

## Troubleshooting

### Common Issues

#### 1. "No module named 'art'"

Install adversarial robustness toolbox:
```bash
pip install adversarial-robustness-toolbox
```

#### 2. CUDA Out of Memory

```python
# Use CPU or smaller batch size
export VISPROBE_DEVICE=cpu
```

#### 3. Tests Never Fail

Your property might be too lenient, or perturbation level too small:

```python
# Increase max level
@search(..., level_hi=1.0)  # Instead of 0.1
```

#### 4. Search Takes Too Long

Reduce query budget or use binary search:

```python
@search(
    mode='binary',  # Faster than adaptive
    max_queries=100  # Limit queries
)
```

#### 5. Normalization Issues

Make sure mean/std match your model's preprocessing:

```python
@data_source(
    data,
    mean=[0.485, 0.456, 0.406],  # ImageNet
    std=[0.229, 0.224, 0.225]
)
```

## Next Steps

- Read the [API Reference](api/index.md) for detailed documentation
- Check out [Examples](examples/index.md) for complete code
- Understand [Design Rationale](design-rationale.md) for advanced usage
- Explore [Architecture](architecture.md) to extend VisProbe
