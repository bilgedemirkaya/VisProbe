# Examples

Working code examples to get you started.

## Quick Start (3 lines)

```python
from visprobe import quick_check
import torchvision.models as models

model = models.resnet18(weights='IMAGENET1K_V1')
test_data = [(torch.randn(3, 224, 224), 0) for _ in range(50)]

report = quick_check(model, test_data, preset="natural")
print(f"Robustness: {report.score:.1%}")
```

## Test Different Threat Models

```python
from visprobe import quick_check

# Test each threat model
report_natural = quick_check(model, data, preset="natural")
report_adv = quick_check(model, data, preset="adversarial")  # Requires ART
report_real = quick_check(model, data, preset="realistic_attack")  # Requires ART

print(f"Natural:     {report_natural.score:.1%}")
print(f"Adversarial: {report_adv.score:.1%}")
print(f"Realistic:   {report_real.score:.1%}")
```

## Compare Models

```python
models_to_test = {
    'resnet18': models.resnet18(weights='DEFAULT'),
    'resnet50': models.resnet50(weights='DEFAULT'),
    'mobilenet': models.mobilenet_v2(weights='DEFAULT'),
}

for name, model in models_to_test.items():
    report = quick_check(model, test_data, preset="natural", budget=500)
    print(f"{name:15} {report.score:.1%}")
```

## Export Hard Cases for Retraining

```python
report = quick_check(model, test_data, preset="natural")

if report.score < 0.75:
    # Export worst cases
    export_path = report.export_failures(n=100)
    print(f"Hard cases exported to {export_path}")

    # Use these in your next training run
    # augmented_data = load_images(export_path)
    # model = retrain(model, augmented_data)
```

## Analyze Failure Patterns

```python
from collections import defaultdict

report = quick_check(model, test_data, preset="natural")

# Group failures by perturbation type
by_strategy = defaultdict(list)
for failure in report.failures:
    by_strategy[failure['strategy']].append(failure)

# Find weakest perturbation
print("Failure counts by strategy:")
for strategy, failures in sorted(by_strategy.items(), key=lambda x: -len(x[1])):
    avg_level = sum(f['level'] for f in failures) / len(failures)
    print(f"  {strategy:20} {len(failures):3d} failures (avg level: {avg_level:.3f})")
```

## Production Validation

```python
def validate_model_before_deployment(model, test_data):
    """Test model meets production requirements."""
    report = quick_check(model, test_data, preset="natural", budget=2000)

    criteria = {
        'robustness': (report.score, 0.75),
        'max_failures': (len(report.failures), 50),
    }

    passed = all(val >= threshold for val, threshold in criteria.values())

    if passed:
        print("✅ Model approved for deployment")
    else:
        print("❌ Model failed validation:")
        for metric, (val, thresh) in criteria.items():
            status = "✅" if val >= thresh else "❌"
            print(f"  {status} {metric}: {val} (threshold: {thresh})")
        report.export_failures(n=20)

    return passed
```

## CI/CD Integration

```python
# In your test file (test_robustness.py)
import pytest
from visprobe import quick_check

def test_model_robustness():
    """CI test: enforce robustness requirements."""
    from your_model import load_model, get_test_data

    model = load_model()
    test_data = get_test_data()

    report = quick_check(model, test_data, preset="natural", budget=1000)

    # Fail if robustness is too low
    assert report.score > 0.70, f"Robustness {report.score:.1%} below 70%"

if __name__ == "__main__":
    test_model_robustness()
```

## Track Robustness Over Time

```python
import json
from datetime import datetime

# After each training run
report = quick_check(model, test_data, preset="natural")

# Save to history
history = []
try:
    with open('robustness_history.json') as f:
        history = json.load(f)
except:
    pass

history.append({
    'date': datetime.now().isoformat(),
    'version': 'v1.2.3',
    'score': report.score,
    'failure_count': len(report.failures)
})

with open('robustness_history.json', 'w') as f:
    json.dump(history, f)

# Print trend
print("Robustness trend:")
for entry in history[-5:]:
    print(f"  {entry['date'][:10]} v{entry['version']:6} Score: {entry['score']:.1%}")
```

## Full Examples on GitHub

For complete, runnable examples with proper setup:

- **[basic_example.py](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples/basic_example.py)** - Minimal usage
- **[cifar10_example.py](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples/cifar10_example.py)** - Full CIFAR-10 workflow
- **[threat_model_comparison.py](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples/comparison.py)** - Compare all threat models
- **[quickstart.ipynb](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples/quickstart.ipynb)** - Interactive notebook

## Need More Help?

- **Getting Started?** See [User Guide](../user-guide.md)
- **API Details?** Check [API Reference](../api/index.md)
- **Still Stuck?** See [Troubleshooting](../TROUBLESHOOTING.md)
