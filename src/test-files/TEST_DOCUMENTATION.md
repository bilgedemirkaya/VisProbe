# VisProbe Test Documentation

This document provides information about the test files in the VisProbe framework, explaining their purpose, strategies, properties, and expected results.

## Overview

The test files in this directory demonstrate different aspects of the VisProbe framework:

- **Adversarial Attacks**: FGSM, PGD, BIM, APGD, Square Attack
- **Natural Perturbations**: Gaussian noise, brightness, rotation
- **Properties**: Label consistency, confidence preservation, top-k stability, output distance
- **Search Modes**: Adaptive, binary, grid, random

## Available Properties

VisProbe provides several robustness properties from the `visprobe.properties` module:

### LabelConstant
Ensures the top-1 prediction remains unchanged after perturbation.

```python
from visprobe.properties import LabelConstant

# Usage in test function
@given(strategy=FGSMStrategy(eps=0.03))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_robustness(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)
```

**Strictness**: Very strict - any label change is a failure

### TopKStability
Checks stability of top-k predictions with multiple modes.

```python
from visprobe.properties import TopKStability

# Mode: overlap - require minimum number of common classes
prop = TopKStability(k=5, mode="overlap", min_overlap=3)
assert prop(original, perturbed)

# Mode: containment - original top-1 must be in perturbed top-k
prop = TopKStability(k=5, mode="containment", require_containment=True)
assert prop(original, perturbed)

# Mode: jaccard - Jaccard similarity between sets
prop = TopKStability(k=5, mode="jaccard", min_jaccard=0.4)
assert prop(original, perturbed)

# Can also use classmethod
TopKStability.evaluate(original, perturbed, k=5, mode="overlap", min_overlap=3)
```

**Modes**:
- `overlap`: Require at least `min_overlap` common classes in top-k sets
- `containment`: Original top-1 must be in perturbed top-k
- `jaccard`: Jaccard index between sets must be ≥ `min_jaccard`

### ConfidenceDrop
Limits confidence decrease in top prediction.

```python
from visprobe.properties import ConfidenceDrop

# Confidence can't drop more than 30%
prop = ConfidenceDrop(max_drop=0.3)
assert prop(original, perturbed)

# Or use classmethod
ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3)
```

**Use case**: Testing confidence preservation under perturbations

### L2Distance
Constrains L2 distance between output logits.

```python
from visprobe.properties import L2Distance

# L2 distance must be <= 1.0
prop = L2Distance(max_delta=1.0)
assert prop(original, perturbed)

# Or use classmethod
L2Distance.evaluate(original, perturbed, max_delta=1.0)
```

**Use case**: Testing output stability at the vector level

## Available Strategies

### Adversarial Attacks

#### FGSMStrategy
Fast Gradient Sign Method - single-step gradient-based attack.

```python
from visprobe.strategies import FGSMStrategy

# Basic usage
strategy = FGSMStrategy(eps=0.03)

# With targeted attack
strategy = FGSMStrategy(eps=0.03, targeted=True)
```

**Parameters**:
- `eps`: Perturbation magnitude (default: 2/255)
- `targeted`: Whether to use targeted attack (default: False)

#### PGDStrategy
Projected Gradient Descent - iterative gradient-based attack.

```python
from visprobe.strategies import PGDStrategy

# Basic usage
strategy = PGDStrategy(eps=0.03, eps_step=0.007, max_iter=40)
```

**Parameters**:
- `eps`: Maximum perturbation
- `eps_step`: Step size per iteration (default: None, auto-calculated)
- `max_iter`: Number of iterations (default: 100)

#### BIMStrategy
Basic Iterative Method - iterated FGSM.

```python
from visprobe.strategies import BIMStrategy

strategy = BIMStrategy(eps=0.03, eps_step=0.007, max_iter=10)
```

**Parameters**:
- `eps`: Maximum perturbation
- `eps_step`: Step size per iteration (default: None, auto-calculated)
- `max_iter`: Number of iterations (default: 10)

#### APGDStrategy
Auto-Projected Gradient Descent.

```python
from visprobe.strategies import APGDStrategy

strategy = APGDStrategy(eps=0.03, max_iter=100)
```

**Parameters**:
- `eps`: Maximum perturbation
- `max_iter`: Number of iterations (default: 100)

#### SquareAttackStrategy
Score-based black-box attack.

```python
from visprobe.strategies import SquareAttackStrategy

strategy = SquareAttackStrategy(eps=0.03, max_iter=5000)
```

**Parameters**:
- `eps`: Maximum perturbation
- `max_iter`: Number of iterations (default: 5000)

### Natural Perturbations

#### GaussianNoiseStrategy
Additive Gaussian noise.

```python
from visprobe.strategies import GaussianNoiseStrategy

# Basic usage
strategy = GaussianNoiseStrategy(std_dev=0.05)

# With custom normalization and seed
strategy = GaussianNoiseStrategy(
    std_dev=0.05,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    seed=42
)
```

**Parameters**:
- `std_dev`: Standard deviation of noise
- `mean`: Normalization mean (optional)
- `std`: Normalization std (optional)
- `seed`: Random seed (optional)

#### BrightnessStrategy
Brightness adjustment.

```python
from visprobe.strategies import BrightnessStrategy

# Increase brightness by 10%
strategy = BrightnessStrategy(brightness_factor=1.1)

# Decrease brightness by 20%
strategy = BrightnessStrategy(brightness_factor=0.8)
```

**Parameters**:
- `brightness_factor`: Brightness multiplier

#### RotateStrategy
Image rotation.

```python
from visprobe.strategies import RotateStrategy

# Rotate by 15 degrees
strategy = RotateStrategy(angle=15)
```

**Parameters**:
- `angle`: Rotation angle in degrees

## Test Patterns

### Basic Fixed Perturbation Test

```python
import visprobe.auto_init
from visprobe import given, model, data_source
from visprobe.strategies import FGSMStrategy
from visprobe.properties import LabelConstant

@given(strategy=FGSMStrategy(eps=0.03))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_fgsm_robustness(original, perturbed):
    """Test robustness against FGSM with single property."""
    assert LabelConstant.evaluate(original, perturbed)
```

### Multi-Property Test

```python
from visprobe.properties import LabelConstant, ConfidenceDrop, TopKStability

@given(strategy=FGSMStrategy(eps=0.03))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_comprehensive_robustness(original, perturbed):
    """Test multiple properties simultaneously."""
    # All conditions must pass
    assert LabelConstant.evaluate(original, perturbed)
    assert ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3)
    assert TopKStability.evaluate(original, perturbed, k=5, mode="overlap", min_overlap=3)
```

### Adaptive Search Test

```python
from visprobe import search

@search(
    strategy=lambda level: FGSMStrategy(eps=level),
    initial_level=0.01,
    step=0.005,
    min_step=0.001,
    max_queries=50,
    mode='adaptive'
)
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def find_failure_threshold(original, perturbed):
    """Find minimum perturbation that causes failure."""
    return LabelConstant.evaluate(original, perturbed)
```

### Binary Search Test (Faster)

```python
@search(
    strategy=lambda level: FGSMStrategy(eps=level),
    mode='binary',
    level_lo=0.0,
    level_hi=0.1,
    min_step=0.001,
    max_queries=30
)
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def find_failure_threshold_fast(original, perturbed):
    """Use binary search for faster threshold discovery."""
    return LabelConstant.evaluate(original, perturbed)
```

### Natural Perturbation Test

```python
from visprobe.strategies import GaussianNoiseStrategy

@given(strategy=GaussianNoiseStrategy(std_dev=0.05))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_noise_robustness(original, perturbed):
    """Test robustness to natural Gaussian noise."""
    # More lenient properties for natural perturbations
    assert ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3)
    assert TopKStability.evaluate(original, perturbed, k=5, mode="overlap", min_overlap=3)
```

### Composite Strategy Test

```python
@given(strategy=[
    GaussianNoiseStrategy(std_dev=0.01),
    BrightnessStrategy(brightness_factor=1.1),
    RotateStrategy(angle=5)
])
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_multiple_perturbations(original, perturbed):
    """Apply multiple perturbations sequentially."""
    assert LabelConstant.evaluate(original, perturbed)
```

## Running Tests

### Run a Test File

```bash
# Run test and save results
visprobe run test_my_model.py

# Run with specific device
visprobe run test_my_model.py --device cuda
visprobe run test_my_model.py --device cpu
```

### Visualize Results

```bash
# Launch interactive dashboard
visprobe visualize test_my_model.py

# Automatically runs test if no results found
visprobe visualize test_my_model.py --device cpu
```

## Search Modes

VisProbe supports four search modes for finding failure thresholds:

### Adaptive (Default)
Step-based search that adapts based on test results.
```python
@search(mode='adaptive', initial_level=0.01, step=0.005)
```

### Binary
True binary search for faster convergence (O(log n)).
```python
@search(mode='binary', level_lo=0.0, level_hi=0.1)
```

### Grid
Systematic grid search over the perturbation space.
```python
@search(mode='grid', level_lo=0.0, level_hi=0.1, step=0.01)
```

### Random
Random sampling within bounds.
```python
@search(mode='random', level_lo=0.0, level_hi=0.1, max_queries=50)
```

## Best Practices

1. **Device Management**: Always use `import visprobe.auto_init` at the top of test files
2. **Property Selection**: Use strict properties (LabelConstant) for adversarial tests, lenient properties (ConfidenceDrop, TopKStability) for natural perturbations
3. **Search Mode**: Use binary search when you need exact thresholds quickly
4. **Multiple Properties**: Test multiple properties for comprehensive evaluation
5. **Documentation**: Add docstrings to test functions explaining the test purpose

## Example Test Files

The `src/test-files/` directory contains various example tests demonstrating:
- Adversarial robustness testing with different attacks
- Natural perturbation testing
- Multi-property evaluation
- Adaptive and binary search
- Composite strategies
- Real-world scenarios

Refer to these examples for practical usage patterns.
