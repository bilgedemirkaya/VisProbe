# VisProbe Documentation

**VisProbe** is a Python library for testing the robustness of vision models against natural perturbations. Test your model's resilience to lighting changes, blur, noise, and more—in just 3 lines of code.

## Overview

VisProbe enables you to:

- **Test robustness in 5 minutes** with a single function call
- **Find failure thresholds** automatically across multiple perturbation types
- **Export failures** for targeted retraining
- **Compare across presets** to identify model weak points
- **Integrate into CI/CD** for continuous robustness monitoring

## Quick Example

```python
import torchvision.models as models
from visprobe import quick_check

# Load your model
model = models.resnet18(weights='IMAGENET1K_V1')

# Test robustness
report = quick_check(model, test_data, preset="standard")

# View results
report.show()
print(f"Robustness Score: {report.score:.1%}")
```

That's it! No decorators, no configuration—just test and get results.

## Key Features

### 1. One-Line API
Get comprehensive robustness testing with a single function call. No boilerplate, no setup.

### 2. Curated Presets
Choose from 4 validated presets designed for different use cases:
- **standard** - General-purpose testing with compositional perturbations
- **lighting** - For outdoor cameras and variable lighting
- **blur** - For video frames and motion handling
- **corruption** - For low-quality inputs and sensor noise

### 3. Compositional Perturbations
Uniquely tests multiple perturbations together (e.g., low-light + blur) to catch real-world failure modes that single perturbations miss.

### 4. Adaptive Search
Automatically finds exact failure thresholds for each perturbation type and sample.

### 5. Actionable Results
- Overall robustness score (0-1)
- List of specific failure cases
- Exportable failure dataset for retraining
- Visual comparison of original vs perturbed images

## Installation

```bash
# From PyPI (when published)
pip install visprobe

# From source
git clone https://github.com/bilgedemirkaya/VisProbe.git
cd VisProbe
pip install -e .
```

## Documentation Structure

- **[User Guide](user-guide.md)** - Getting started and common workflows
- **[Architecture](architecture.md)** - System design and components
- **[Design Rationale](design-rationale.md)** - Why we made certain choices
- **[Examples](../examples/README.md)** - Complete working examples
- **[API Reference](../COMPREHENSIVE_API_REFERENCE.md)** - Detailed API documentation
- **[Troubleshooting](../TROUBLESHOOTING.md)** - Common issues and solutions

## Use Cases

### 1. Production Deployment Validation

```python
from visprobe import quick_check
import your_model

def test_production_robustness():
    model = your_model.load_production_checkpoint()
    test_data = your_model.get_test_data()

    report = quick_check(model, test_data, preset="standard", budget=1000)

    # Enforce robustness requirement
    assert report.score > 0.70, f"Model too fragile: {report.score:.1%}"

    return report
```

### 2. Model Comparison

```python
# Compare two models
report_v1 = quick_check(model_v1, test_data, preset="standard")
report_v2 = quick_check(model_v2, test_data, preset="standard")

print(f"v1 score: {report_v1.score:.1%}")
print(f"v2 score: {report_v2.score:.1%}")
print(f"Improvement: {(report_v2.score - report_v1.score)*100:.1f}%")
```

### 3. Targeted Retraining

```python
# Find failures and export for retraining
report = quick_check(model, test_data, preset="standard")

if report.score < 0.80:
    failures_path = report.export_failures(n=50)
    print(f"Add {failures_path} to your training set")
```

### 4. CI/CD Integration

```bash
# In your CI pipeline
python -c "
from visprobe import quick_check
import your_model

report = quick_check(
    your_model.load(),
    your_model.get_test_data(),
    preset='standard'
)
assert report.score > 0.70
"
```

## Quick Links

- **[Main README](../README.md)** - Project overview
- **[Examples](../examples/)** - Working code examples
- **[GitHub Repository](https://github.com/bilgedemirkaya/VisProbe)** - Source code
- **[Report Issues](https://github.com/bilgedemirkaya/VisProbe/issues)** - Bug reports

## Citation

If you use VisProbe in your research, please cite:

```bibtex
@software{visprobe,
  title={VisProbe: Robustness Testing for Vision Models},
  author={Bilge Demirkaya},
  year={2025},
  url={https://github.com/bilgedemirkaya/VisProbe}
}
```

## Next Steps

- **New to VisProbe?** Start with the [Quick Start Guide](user-guide.md)
- **Have your own model?** See [examples/custom_model_example.py](../examples/custom_model_example.py)
- **Want to compare presets?** See [examples/preset_comparison.py](../examples/preset_comparison.py)
- **Need help?** Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
