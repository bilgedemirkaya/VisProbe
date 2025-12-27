# User Guide

Get started testing your model's robustness.

## Installation

```bash
pip install visprobe

# For adversarial testing (optional)
pip install adversarial-robustness-toolbox
```

## 5-Minute Quick Start

```python
import torch
import torchvision.models as models
from visprobe import quick_check

# Load model
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# Prepare data (list of tuples or DataLoader)
test_data = [(image1, label1), (image2, label2), ...]

# Run test
report = quick_check(model, test_data, preset="natural")

# View results
print(f"Score: {report.score:.1%}")
report.show()  # Rich interactive display
```

## Presets: Choose What to Test

| Preset | Tests | Time | When to Use |
|--------|-------|------|-------------|
| `natural` | Brightness, blur, noise, compression | 12-15 min | Most use cases |
| `adversarial` | FGSM, PGD, BIM attacks | 15-25 min | Adversarial robustness |
| `realistic_attack` | Attacks in real conditions ⭐ | 20-30 min | Security-critical systems |
| `comprehensive` | All three threat models | 45-60 min | Research / benchmarking |

**Most users should start with `natural` or `realistic_attack`.**

## Understanding Results

```python
# Overall score (0-1)
print(report.score)  # 0.75 = 75% robust

# Per-threat-model breakdown
print(report.threat_model_scores)
# {'natural': 0.80, 'adversarial': 0.65, 'realistic_attack': 0.55}

# List failures
for failure in report.failures[:5]:
    print(f"Failed on {failure['strategy']} at level {failure['level']:.3f}")

# Export hard cases for retraining
report.export_failures(n=50, output_dir="./hard_cases")
```

## Common Tasks

### Task 1: Test a New Model

```python
from visprobe import quick_check

report = quick_check(model, test_data, preset="natural", budget=1000)

if report.score > 0.75:
    print("✅ Model is robust enough")
else:
    print("⚠️ Model needs robustness work")
    report.export_failures(n=50)  # Export to retrain on
```

### Task 2: Compare Two Models

```python
models = {'v1': model_v1, 'v2': model_v2}

for name, model in models.items():
    report = quick_check(model, test_data, preset="natural")
    print(f"{name}: {report.score:.1%}")
```

### Task 3: Find Weak Points

```python
# Test all presets to find where model struggles most
results = {}
for preset in ["natural", "lighting", "blur", "corruption"]:
    report = quick_check(model, test_data, preset=preset, budget=500)
    results[preset] = report.score

weakest = min(results.items(), key=lambda x: x[1])
print(f"Weakest: {weakest[0]} ({weakest[1]:.1%})")
```

### Task 4: CI/CD Integration

```python
# In your test suite
def test_model_robustness():
    report = quick_check(model, test_data, preset="natural")
    assert report.score > 0.70, f"Robustness too low: {report.score:.1%}"
```

## Advanced Configuration

### Custom Normalization

If your model expects normalized inputs:

```python
report = quick_check(
    model,
    test_data,
    preset="natural",
    mean=(0.485, 0.456, 0.406),  # ImageNet defaults
    std=(0.229, 0.224, 0.225)
)
```

**Note:** Only pass mean/std if data isn't already normalized.

### Adjust Budget vs Speed

```python
# Fast (development)
report = quick_check(model, data, preset="natural", budget=100)

# Normal (recommended)
report = quick_check(model, data, preset="natural", budget=1000)

# Precise (production)
report = quick_check(model, data, preset="natural", budget=5000)
```

### Force Device

```python
report = quick_check(model, data, device="cuda")  # Force GPU
report = quick_check(model, data, device="cpu")   # Force CPU
```

## Interpreting Scores

- **> 0.80** ✅ Excellent - Model is production-ready
- **0.60-0.80** ⚠️ Good - Reasonable robustness
- **0.40-0.60** ❌ Moderate - Needs improvement
- **< 0.40** ❌ Poor - Requires significant work

## Best Practices

1. **Start small** - Test on 10-20 samples first
2. **Match your training** - Use same normalization as training
3. **Use realistic data** - Test on real test set, not random noise
4. **Track over time** - Save reports to track improvements
5. **Export failures** - Use them in next training iteration

## Troubleshooting

**Out of memory?**
```python
# Reduce batch size internally (budget)
report = quick_check(model, data, budget=100)

# Or force CPU
report = quick_check(model, data, device="cpu")
```

**Model not loading?**
```python
# Make sure model is on correct device
model = model.to('cuda')
model.eval()  # Critical: disable dropout, batch norm
```

**Getting weird results?**
```python
# Verify normalization matches training
# Verify test data format is correct
# Check that labels are in [0, num_classes)
```

## Next Steps

- **API Details** - See [API Reference](COMPREHENSIVE_API_REFERENCE.md)
- **More Help** - Check [Troubleshooting](TROUBLESHOOTING.md)
- **Code Examples** - See [examples on GitHub](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples)
