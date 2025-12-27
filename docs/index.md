# VisProbe Documentation

**VisProbe** makes it easy to test how robust your vision models are. Test against natural effects, adversarial attacks, and real-world combinations in just 3 lines of code.

## Why VisProbe?

Most robustness testing is incomplete. Tools test:
- ✓ Can your model handle brightness changes?
- ✓ Can it resist FGSM attacks?
- ✗ Can it resist FGSM in low-light? (the real attack!)

VisProbe tests the complete picture with three threat models:
- **Natural** - Environmental effects (brightness, blur, noise)
- **Adversarial** - Gradient-based attacks on clean images
- **Realistic** - Attacks under real conditions (low-light + adversarial) ⭐

## Quick Start

```python
from visprobe import quick_check
import torchvision.models as models

model = models.resnet18(weights='DEFAULT')
test_data = [(image, label), ...]  # Your test data

report = quick_check(model, test_data, preset="natural")
print(f"Robustness: {report.score:.1%}")
```

That's it! VisProbe automatically finds failure thresholds for each perturbation.

## Key Features

**Threat-Model-Aware Presets**
- `natural` - Environmental testing (12-15 min)
- `adversarial` - Adversarial attacks (15-25 min)
- `realistic_attack` - Real-world attacks (20-30 min) ⭐ Use this!
- `comprehensive` - All three at once (45-60 min)

**Get Actionable Results**
```python
report.show()                    # Rich interactive display
report.export_failures(n=20)     # Export hard cases for retraining
print(report.threat_model_scores) # See scores per threat model
```

**Integrate Anywhere**
```python
# CI/CD pipelines
assert report.score > 0.70

# Production monitoring
if report.vulnerability_warning:
    trigger_alert()
```

## Common Use Cases

### Testing a New Model
```python
report = quick_check(model, test_data, preset="natural")
if report.score < 0.75:
    report.export_failures(n=50)  # Get hard cases to fix
```

### Comparing Model Versions
```python
for model in [model_v1, model_v2, model_v3]:
    report = quick_check(model, data, preset="natural")
    print(f"Version: {report.score:.1%}")
```

### Production Validation
```python
report = quick_check(model, prod_test_data, preset="comprehensive")
criteria = {
    'natural': 0.75,
    'adversarial': 0.60,
    'realistic': 0.50
}
for threat_model, threshold in criteria.items():
    assert report.threat_model_scores[threat_model] > threshold
```

## Next Steps

- **Getting started?** See the [User Guide](user-guide.md)
- **Want details?** Check the [API Reference](COMPREHENSIVE_API_REFERENCE.md)
- **Need help?** Read [Troubleshooting](TROUBLESHOOTING.md)
- **Run examples?** Check [GitHub examples](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples)

---

**Latest:** Updated for threat-model-aware presets (VisProbe 2.0)
