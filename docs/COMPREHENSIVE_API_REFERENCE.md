# VisProbe: Complete API Reference

This document provides a comprehensive reference for the VisProbe API.

## Table of Contents

1. [Quick Check API](#quick-check-api)
2. [Report Object](#report-object)
3. [Presets](#presets)
4. [Strategies](#strategies)
5. [Properties](#properties)
6. [Advanced Usage](#advanced-usage)

---

## Quick Check API

### `quick_check()`

The primary function for testing model robustness.

```python
from visprobe import quick_check

report = quick_check(
    model,
    data,
    preset="standard",
    budget=1000,
    device="auto",
    output_dir="visprobe_results",
    mean=None,
    std=None
)
```

**Parameters:**

- **`model`** (required): Your PyTorch model (nn.Module) or any callable that takes tensors and returns logits
- **`data`** (required): Test data in any of these formats:
  - `DataLoader` object
  - `TensorDataset` object
  - List of `(image, label)` tuples
  - Tuple of `(images_tensor, labels_tensor)`
- **`preset`** (str, default="standard"): Which preset to use
  - `"standard"` - Balanced mix including compositional perturbations
  - `"lighting"` - Brightness, contrast, gamma variations
  - `"blur"` - Gaussian blur, motion blur, JPEG compression
  - `"corruption"` - Noise, artifacts, degradation
- **`budget`** (int, default=1000): Maximum number of model queries per strategy
  - Higher budget = more precise failure thresholds
  - Recommended: 500-1000 for most use cases
- **`device`** (str, default="auto"): Device to run on
  - `"auto"` - Auto-detect (CUDA → MPS → CPU)
  - `"cuda"` - Force CUDA GPU
  - `"mps"` - Force Apple Silicon GPU
  - `"cpu"` - Force CPU
- **`output_dir`** (str, default="visprobe_results"): Where to save results
- **`mean`** (tuple, optional): Normalization mean (e.g., `(0.485, 0.456, 0.406)`)
  - If None, assumes data is already normalized or in [0, 1]
- **`std`** (tuple, optional): Normalization std dev (e.g., `(0.229, 0.224, 0.225)`)
  - If None, assumes data is already normalized or in [0, 1]

**Returns:**

A `Report` object with test results.

**Example:**

```python
import torch
import torchvision.models as models
from visprobe import quick_check

# Load model
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# Prepare test data
test_data = [(torch.randn(3, 224, 224), 0) for _ in range(50)]

# Test robustness
report = quick_check(
    model=model,
    data=test_data,
    preset="standard",
    budget=1000,
    device="auto"
)

# View results
report.show()
print(f"Robustness Score: {report.score:.1%}")
```

---

## Report Object

The `Report` object contains all test results and provides convenient access methods.

### Properties

#### `.score`

Overall robustness score (0-1, higher is better).

```python
score = report.score  # e.g., 0.847
print(f"Robustness: {score:.1%}")  # "Robustness: 84.7%"
```

**Interpretation:**
- **> 0.80** - Excellent robustness
- **0.60-0.80** - Good robustness
- **0.40-0.60** - Moderate robustness issues
- **< 0.40** - Poor robustness

#### `.failures`

List of failure cases found during testing.

```python
failures = report.failures
print(f"Found {len(failures)} failures")

for failure in failures[:5]:
    print(f"Sample {failure['index']}: {failure['original_pred']} → {failure['perturbed_pred']}")
    print(f"  Strategy: {failure['strategy']}, Level: {failure['level']:.3f}")
```

**Failure Dict Structure:**
```python
{
    'index': int,              # Sample index in dataset
    'original_pred': int,      # Original predicted class
    'perturbed_pred': int,     # Prediction after perturbation
    'strategy': str,           # Strategy that caused failure
    'level': float,            # Perturbation level
    'original_image': Tensor,  # Original image
    'perturbed_image': Tensor  # Perturbed image
}
```

#### `.summary`

Dictionary with key metrics.

```python
summary = report.summary
print(summary)
```

**Keys:**
- `overall_robustness_score`: Float (0-1)
- `total_samples`: Int
- `passed_samples`: Int
- `failed_samples`: Int
- `num_strategies`: Int
- `runtime_sec`: Float
- `model_queries`: Int
- `preset_used`: Str

### Methods

#### `.show(mode=None)`

Display results in a context-appropriate way.

```python
# In Jupyter: Shows rich HTML
report.show()

# In interactive Python: Shows colored text
report.show()

# Force specific mode
report.show(mode="html")    # Force HTML
report.show(mode="text")    # Force plain text
```

**Auto-detection:**
- Jupyter notebook → Rich HTML with charts
- Interactive terminal → Colored text output
- Script/non-interactive → Plain text

#### `.export_failures(n=10, output_dir=None)`

Export the worst failure cases as a dataset for retraining.

```python
# Export top 10 failures
path = report.export_failures(n=10)
print(f"Exported to: {path}")

# Custom output directory
path = report.export_failures(n=20, output_dir="./hard_examples")
```

**Returns:** Path to the exported dataset directory

**Output Structure:**
```
visprobe_results/failures_export/
├── manifest.json          # Metadata about failures
├── 0_original.png        # Original image
├── 0_perturbed.png       # Perturbed version
├── 1_original.png
├── 1_perturbed.png
└── ...
```

**Usage in Training:**
```python
# Load exported failures
import json
with open('visprobe_results/failures_export/manifest.json') as f:
    failures = json.load(f)

# Add to training set
for failure in failures:
    training_dataset.add_hard_example(
        image_path=failure['perturbed_image_path'],
        label=failure['original_label']
    )
```

#### `.save(path=None)`

Save the full report to disk.

```python
# Save to default location
report.save()

# Save to custom path
report.save("my_experiment_results.json")
```

---

## Presets

Presets are curated bundles of perturbation strategies with validated parameter ranges.

### Available Presets

#### "standard"

Balanced mix for general-purpose robustness testing.

**Strategies:**
- Brightness (0.6-1.4x)
- Gaussian blur (σ: 0-2.5)
- Gaussian noise (std: 0-0.05)
- JPEG compression (quality: 10-100)
- **Compositional perturbations:**
  - Low-light + blur
  - Compression + noise

**Best for:** Production models, general robustness testing

**Example:**
```python
report = quick_check(model, data, preset="standard")
```

#### "lighting"

Tests robustness to lighting variations.

**Strategies:**
- Brightness (0.5-1.5x)
- Contrast (0.7-1.3x)
- Gamma correction (0.7-1.3)
- **Compositional:** Dim low-contrast

**Best for:** Outdoor cameras, variable lighting conditions

**Example:**
```python
report = quick_check(model, data, preset="lighting")
```

#### "blur"

Tests robustness to blur and compression.

**Strategies:**
- Gaussian blur (σ: 0-3.0)
- Motion blur (kernel: 1-25 pixels)
- JPEG compression (quality: 10-100)

**Best for:** Video frames, compressed images, motion handling

**Example:**
```python
report = quick_check(model, data, preset="blur")
```

#### "corruption"

Tests robustness to noise and degradation.

**Strategies:**
- Gaussian noise (std: 0-0.08)
- JPEG compression (quality: 5-100)
- **Compositional:** Compression + noise

**Best for:** Low-quality inputs, sensor noise, transmission errors

**Example:**
```python
report = quick_check(model, data, preset="corruption")
```

### Listing Presets Programmatically

```python
from visprobe import presets

# List all available presets
for name, description in presets.list_presets():
    print(f"{name}: {description}")

# Get preset details
preset_config = presets.get_preset("standard")
print(preset_config['name'])
print(preset_config['description'])
print(preset_config['strategies'])
```

---

## Strategies

Strategies define how images are perturbed. While presets bundle strategies together, you can also use individual strategies for custom testing.

### Natural Perturbation Strategies

#### BrightnessStrategy

Adjusts image brightness.

```python
from visprobe.strategies.image import BrightnessStrategy

strategy = BrightnessStrategy(brightness_factor=1.2)
```

**Parameters:**
- `brightness_factor` (float): Brightness multiplier
  - < 1.0: Darker
  - = 1.0: No change
  - > 1.0: Brighter

#### ContrastStrategy

Adjusts image contrast.

```python
from visprobe.strategies.image import ContrastStrategy

strategy = ContrastStrategy(contrast_factor=1.3)
```

**Parameters:**
- `contrast_factor` (float): Contrast multiplier
  - < 1.0: Lower contrast
  - = 1.0: No change
  - > 1.0: Higher contrast

#### GammaStrategy

Applies gamma correction.

```python
from visprobe.strategies.image import GammaStrategy

strategy = GammaStrategy(gamma=1.2, gain=1.0)
```

**Parameters:**
- `gamma` (float): Gamma value
  - < 1.0: Brighten
  - = 1.0: No change
  - > 1.0: Darken
- `gain` (float, default=1.0): Gain multiplier

#### GaussianBlurStrategy

Applies Gaussian blur.

```python
from visprobe.strategies.image import GaussianBlurStrategy

strategy = GaussianBlurStrategy(kernel_size=5, sigma=2.0)
```

**Parameters:**
- `kernel_size` (int or tuple): Blur kernel size (must be odd)
- `sigma` (float or tuple): Standard deviation
  - Higher sigma = more blur

#### MotionBlurStrategy

Simulates motion blur.

```python
from visprobe.strategies.image import MotionBlurStrategy

strategy = MotionBlurStrategy(kernel_size=15, angle=45.0)
```

**Parameters:**
- `kernel_size` (int): Size of motion blur kernel
- `angle` (float): Direction of motion in degrees

#### JPEGCompressionStrategy

Simulates JPEG compression artifacts.

```python
from visprobe.strategies.image import JPEGCompressionStrategy

strategy = JPEGCompressionStrategy(quality=75)
```

**Parameters:**
- `quality` (int): JPEG quality (1-100)
  - Lower = more artifacts

#### GaussianNoiseStrategy

Adds Gaussian noise.

```python
from visprobe.strategies.image import GaussianNoiseStrategy

strategy = GaussianNoiseStrategy(std_dev=0.05, seed=42)
```

**Parameters:**
- `std_dev` (float): Standard deviation of noise
- `mean`, `std` (optional): Denormalization parameters
- `seed` (optional): Random seed for reproducibility

---

## Properties

Properties define what constitutes a "failure". The default property used by `quick_check()` is `LabelConstant`, which checks if the predicted class changes.

### LabelConstant

Checks if the top-1 prediction remains the same.

```python
from visprobe.properties import LabelConstant

prop = LabelConstant()
passed = prop(original_logits, perturbed_logits)
```

**Returns:** `True` if top-1 prediction is unchanged, `False` otherwise.

### TopKStability

Checks stability of top-k predictions.

```python
from visprobe.properties import TopKStability

# Overlap mode: At least 3 common classes in top-5
prop = TopKStability(k=5, mode="overlap", min_overlap=3)

# Containment mode: Original top-1 must be in perturbed top-k
prop = TopKStability(k=5, mode="containment")

# Jaccard mode: Jaccard similarity ≥ 0.4
prop = TopKStability(k=5, mode="jaccard", min_jaccard=0.4)
```

### ConfidenceDrop

Limits the allowed confidence decrease.

```python
from visprobe.properties import ConfidenceDrop

# Allow max 30% confidence drop
prop = ConfidenceDrop(max_drop=0.3)
```

### L2Distance

Constrains L2 distance between logits.

```python
from visprobe.properties import L2Distance

# Max L2 distance of 1.0
prop = L2Distance(max_delta=1.0)
```

---

## Advanced Usage

### Custom Normalization

If your model was trained with specific normalization:

```python
# ImageNet normalization
report = quick_check(
    model,
    data,
    preset="standard",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

### Multiple Presets

Compare your model across all presets:

```python
results = {}
for preset_name in ["standard", "lighting", "blur", "corruption"]:
    report = quick_check(model, data, preset=preset_name, budget=500)
    results[preset_name] = report.score

# Find weakest area
weakest = min(results.items(), key=lambda x: x[1])
print(f"Weakest area: {weakest[0]} (score: {weakest[1]:.1%})")
```

### Production CI/CD Integration

Use VisProbe in automated testing:

```python
# test_model_robustness.py
import torch
from visprobe import quick_check
from my_project import load_model, get_test_data

def test_production_robustness():
    """Ensure model meets robustness requirements."""
    model = load_model("production_checkpoint.pth")
    test_data = get_test_data(num_samples=100)

    report = quick_check(
        model,
        test_data,
        preset="standard",
        budget=1000,
        device="cuda"
    )

    # Enforce minimum robustness score
    assert report.score > 0.70, \
        f"Model robustness too low: {report.score:.1%} (required: >70%)"

    # Save report for tracking
    report.save(f"robustness_report_{datetime.now()}.json")

    return report

if __name__ == "__main__":
    test_production_robustness()
```

### Custom Test Data

```python
from torch.utils.data import DataLoader, TensorDataset

# From DataLoader
loader = DataLoader(my_dataset, batch_size=32)
report = quick_check(model, loader, preset="standard")

# From TensorDataset
dataset = TensorDataset(images_tensor, labels_tensor)
report = quick_check(model, dataset, preset="standard")

# From raw tensors
images = torch.randn(100, 3, 224, 224)
labels = torch.randint(0, 1000, (100,))
report = quick_check(model, (images, labels), preset="standard")
```

### Analyzing Failure Patterns

```python
report = quick_check(model, data, preset="standard")

# Group failures by strategy
from collections import defaultdict
by_strategy = defaultdict(list)
for failure in report.failures:
    by_strategy[failure['strategy']].append(failure)

# Find most problematic strategy
for strategy, failures in by_strategy.items():
    print(f"{strategy}: {len(failures)} failures")
    avg_level = sum(f['level'] for f in failures) / len(failures)
    print(f"  Average failure level: {avg_level:.3f}")
```

---

## Best Practices

1. **Start with small budget for quick iteration**
   ```python
   # Quick test during development
   report = quick_check(model, data[:10], preset="standard", budget=100)
   ```

2. **Use appropriate preset for your domain**
   - Outdoor cameras → "lighting"
   - Video processing → "blur"
   - Low-quality inputs → "corruption"
   - General purpose → "standard"

3. **Set normalization to match training**
   ```python
   report = quick_check(
       model, data, preset="standard",
       mean=YOUR_TRAINING_MEAN,
       std=YOUR_TRAINING_STD
   )
   ```

4. **Monitor robustness in CI/CD**
   ```python
   assert report.score > MINIMUM_THRESHOLD
   ```

5. **Export and retrain on failures**
   ```python
   if report.score < 0.80:
       report.export_failures(n=50)
       # Add to training set and retrain
   ```

6. **Track robustness over time**
   ```python
   report.save(f"robustness_{model_version}.json")
   # Compare with previous versions
   ```

---

## Environment Variables

- `VISPROBE_DEVICE`: Force specific device (cpu, cuda, mps)
- `VISPROBE_DEBUG`: Enable debug output

Example:
```bash
export VISPROBE_DEVICE=cpu
python test_my_model.py
```

---

## Getting Help

- **Documentation**: See [README_ROOT.md](README_ROOT.md)
- **Examples**: See the examples on [GitHub](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples)
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: Report bugs on GitHub

---

This API reference covers all public interfaces in VisProbe. For implementation details, see the source code.
