# Examples

**For complete working examples, see:** [examples/](../../examples/)

## Quick Examples

### 1. Basic Usage (3 lines!)

```python
from visprobe import quick_check
import torchvision.models as models

model = models.resnet18(weights='IMAGENET1K_V1')
test_data = [(torch.randn(3, 224, 224), 0) for _ in range(50)]

report = quick_check(model, test_data, preset="standard")
print(f"Robustness: {report.score:.1%}")
```

### 2. CIFAR-10 Example

```python
from visprobe import quick_check
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

# Load data
transform = T.Compose([T.Resize(224), T.ToTensor()])
dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_data = [dataset[i] for i in range(100)]

# Test robustness
report = quick_check(model, test_data, preset="standard", budget=1000)

# Analyze results
report.show()
if report.score < 0.70:
    report.export_failures(n=20)
```

### 3. Compare Presets

```python
from visprobe import quick_check

presets = ["standard", "lighting", "blur", "corruption"]
results = {}

for preset_name in presets:
    report = quick_check(model, test_data, preset=preset_name, budget=500)
    results[preset_name] = report.score
    print(f"{preset_name}: {report.score:.1%}")

# Find weakest area
weakest = min(results.items(), key=lambda x: x[1])
print(f"\nWeakest area: {weakest[0]}")
```

### 4. Custom Normalization

```python
from visprobe import quick_check

# ImageNet normalization
report = quick_check(
    model,
    test_data,
    preset="standard",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
```

### 5. Export and Analyze Failures

```python
report = quick_check(model, test_data, preset="standard")

# Export worst failures
export_path = report.export_failures(n=50)
print(f"Exported to: {export_path}")

# Analyze by strategy
from collections import defaultdict

by_strategy = defaultdict(list)
for failure in report.failures:
    by_strategy[failure['strategy']].append(failure)

for strategy, failures in by_strategy.items():
    print(f"{strategy}: {len(failures)} failures")
```

## Complete Examples

For full, runnable examples:

### [examples/basic_example.py](../../examples/basic_example.py)
Minimal 3-line example showing the simplest possible usage.

**Run time:** ~30 seconds

### [examples/cifar10_example.py](../../examples/cifar10_example.py)
Complete CIFAR-10 workflow with proper normalization, failure analysis, and export.

**Run time:** ~5-10 minutes

### [examples/custom_model_example.py](../../examples/custom_model_example.py)
Template you can copy and modify for your own model.

**Includes:**
- Data loading
- Preset selection
- Result interpretation
- Best practices

### [examples/preset_comparison.py](../../examples/preset_comparison.py)
Compare all 4 presets to identify model weak points.

**Output:**
- Per-preset scores
- Weakest area identification
- Recommendations

## Jupyter Notebook

For an interactive walkthrough:

### [examples/quickstart.ipynb](../../examples/quickstart.ipynb)

**Covers:**
- Installation
- Basic usage
- Result interpretation
- Failure analysis
- Export and retraining
- Preset comparison

**Time to complete:** 10-15 minutes

## Use Case Examples

### Production Validation

```python
def test_production_robustness():
    model = load_model("production_checkpoint.pth")
    test_data = load_test_set()

    report = quick_check(model, test_data, preset="standard", budget=1000)

    # Enforce threshold
    assert report.score > 0.70, f"Robustness too low: {report.score:.1%}"

    report.save(f"robustness_report_{version}.json")
    return report
```

### CI/CD Integration

```python
# In your test suite
def test_model_robustness():
    """CI test: model meets robustness requirements."""
    report = quick_check(model, test_data, preset="standard")
    assert report.score > THRESHOLD
```

### Targeted Retraining

```python
report = quick_check(model, test_data, preset="standard")

if report.score < 0.80:
    # Export failures
    path = report.export_failures(n=100)

    # Add to training set
    add_hard_examples(path)

    # Retrain
    retrain_model()
```

## Next Steps

- **Try the examples:** Run `python examples/basic_example.py`
- **Read the guide:** See [User Guide](../user-guide.md)
- **Full API reference:** See [API Reference](../../COMPREHENSIVE_API_REFERENCE.md)
- **Troubleshooting:** See [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
