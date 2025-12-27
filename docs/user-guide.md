# VisProbe User Guide

This guide will help you get started with VisProbe and test the robustness of your vision models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Choosing a Preset](#choosing-a-preset)
5. [Understanding Results](#understanding-results)
6. [Working with Failures](#working-with-failures)
7. [Advanced Configuration](#advanced-configuration)
8. [Production Integration](#production-integration)
9. [Best Practices](#best-practices)
10. [Common Workflows](#common-workflows)

---

## Installation

### Basic Installation

```bash
# From PyPI (when published)
pip install visprobe

# From source
git clone https://github.com/bilgedemirkaya/VisProbe.git
cd VisProbe
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision

VisProbe will automatically install all required dependencies.

---

## Quick Start

### 1. Import and Load Model

```python
import torch
import torchvision.models as models
from visprobe import quick_check

# Load your model
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()
```

### 2. Prepare Test Data

```python
# Option 1: List of (image, label) tuples
test_data = [(image1, label1), (image2, label2), ...]

# Option 2: From a DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)
test_data = loader

# Option 3: Raw tensors
images = torch.randn(50, 3, 224, 224)
labels = torch.randint(0, 1000, (50,))
test_data = (images, labels)
```

### 3. Run Robustness Test

```python
report = quick_check(
    model=model,
    data=test_data,
    preset="standard",
    budget=1000,
    device="auto"
)
```

### 4. View Results

```python
# Interactive display (context-aware)
report.show()

# Access metrics programmatically
print(f"Robustness Score: {report.score:.1%}")
print(f"Total Failures: {len(report.failures)}")
print(f"Runtime: {report.summary['runtime_sec']:.1f}s")
```

**That's it!** You've tested your model's robustness in just a few lines.

---

## Core Concepts

### 1. The `quick_check()` Function

The primary interface to VisProbe. It:

- Tests your model against a preset of perturbations
- Automatically finds failure thresholds for each perturbation
- Returns a comprehensive Report object

**Signature:**
```python
quick_check(
    model,           # Your PyTorch model
    data,            # Test dataset
    preset="standard",  # Which preset to use
    budget=1000,     # Model queries per strategy
    device="auto",   # "auto", "cuda", "cpu", or "mps"
    mean=None,       # Normalization mean (if needed)
    std=None         # Normalization std (if needed)
)
```

### 2. Presets

Presets are curated bundles of perturbations with validated parameter ranges. They're designed for specific use cases:

- **standard** - General-purpose testing (recommended for most users)
- **lighting** - Brightness, contrast, gamma variations
- **blur** - Blur and compression effects
- **corruption** - Noise and degradation

Each preset includes compositional perturbations that test multiple effects together.

### 3. The Report Object

Contains all test results and provides convenient methods:

**Properties:**
- `.score` - Overall robustness score (0-1)
- `.failures` - List of failure cases
- `.summary` - Dictionary of key metrics

**Methods:**
- `.show()` - Display results (context-aware)
- `.export_failures(n)` - Export worst failures
- `.save(path)` - Save report to disk

### 4. Adaptive Search

For each perturbation type, VisProbe automatically:

1. Starts with the minimum perturbation level
2. Gradually increases the perturbation
3. Finds the exact threshold where the model fails
4. Tracks failures for each test sample

This gives you precise understanding of your model's limits.

---

## Choosing a Preset

### "standard" - General Purpose ✅

**Best for:** Most production models, general robustness testing

**Includes:**
- Brightness (0.6-1.4x)
- Gaussian blur (σ: 0-2.5)
- Gaussian noise (std: 0-0.05)
- JPEG compression (quality: 10-100)
- Compositional: Low-light + blur
- Compositional: Compression + noise

**When to use:**
- First-time testing
- General production deployment
- Comparing model versions
- Not sure which preset to pick

**Example:**
```python
report = quick_check(model, data, preset="standard")
```

### "lighting" - Variable Lighting ☀️

**Best for:** Outdoor cameras, variable lighting conditions

**Includes:**
- Brightness (0.5-1.5x)
- Contrast (0.7-1.3x)
- Gamma correction (0.7-1.3)
- Compositional: Dim low-contrast scenes

**When to use:**
- Outdoor deployment (day/night variations)
- Indoor with variable lighting
- Security cameras
- Autonomous vehicles

**Example:**
```python
report = quick_check(model, data, preset="lighting")
```

### "blur" - Motion and Compression 🎥

**Best for:** Video processing, compressed images

**Includes:**
- Gaussian blur (σ: 0-3.0)
- Motion blur (kernel: 1-25 pixels)
- JPEG compression (quality: 10-100)

**When to use:**
- Video frame analysis
- Real-time processing
- Motion-heavy scenarios
- Compressed image streams

**Example:**
```python
report = quick_check(model, data, preset="blur")
```

### "corruption" - Noise and Degradation 📡

**Best for:** Low-quality inputs, sensor noise

**Includes:**
- Gaussian noise (std: 0-0.08)
- JPEG compression (quality: 5-100)
- Compositional: Compression + noise

**When to use:**
- Low-quality cameras
- Sensor noise
- Transmission errors
- Medical imaging

**Example:**
```python
report = quick_check(model, data, preset="corruption")
```

---

## Understanding Results

### Robustness Score

The `.score` property gives an overall robustness rating from 0 to 1:

```python
score = report.score

if score > 0.80:
    print("✅ Excellent - Model is highly robust")
elif score > 0.60:
    print("✅ Good - Reasonable robustness")
elif score > 0.40:
    print("⚠️ Moderate - Significant robustness issues")
else:
    print("❌ Poor - Model is very fragile")
```

**What it means:**
- **> 0.80** - Most samples survive perturbations, model is production-ready
- **0.60-0.80** - Some weaknesses but generally acceptable
- **0.40-0.60** - Considerable failures, needs improvement
- **< 0.40** - Model struggles badly, requires robustness work

### Failure Analysis

The `.failures` property lists all failure cases:

```python
for failure in report.failures[:10]:
    print(f"Sample {failure['index']}:")
    print(f"  Original: class {failure['original_pred']}")
    print(f"  Perturbed: class {failure['perturbed_pred']}")
    print(f"  Strategy: {failure['strategy']}")
    print(f"  Level: {failure['level']:.3f}")
    print()
```

**Each failure includes:**
- `index` - Which test sample failed
- `original_pred` - What the model predicted originally
- `perturbed_pred` - What it predicted after perturbation
- `strategy` - Which perturbation caused the failure
- `level` - How much perturbation was needed
- `original_image` & `perturbed_image` - The actual images

### Summary Metrics

The `.summary` property provides aggregate statistics:

```python
summary = report.summary
print(f"Total samples tested: {summary['total_samples']}")
print(f"Samples that passed: {summary['passed_samples']}")
print(f"Samples that failed: {summary['failed_samples']}")
print(f"Number of strategies: {summary['num_strategies']}")
print(f"Runtime: {summary['runtime_sec']:.1f}s")
print(f"Model queries used: {summary['model_queries']}")
```

---

## Working with Failures

### Exporting for Retraining

Export the worst failures to use as hard examples in training:

```python
report = quick_check(model, data, preset="standard")

if report.score < 0.80:
    # Export top 50 failures
    export_path = report.export_failures(n=50)
    print(f"Exported to: {export_path}")

    # Load manifest
    import json
    with open(f"{export_path}/manifest.json") as f:
        failures = json.load(f)

    # Add to training set
    for failure in failures:
        training_set.add_hard_example(
            image=failure['perturbed_image_path'],
            label=failure['original_label']
        )
```

### Analyzing Patterns

Group failures by strategy to find weak points:

```python
from collections import defaultdict

by_strategy = defaultdict(list)
for failure in report.failures:
    by_strategy[failure['strategy']].append(failure)

print("Failures by strategy:")
for strategy, failures in sorted(by_strategy.items(), key=lambda x: -len(x[1])):
    avg_level = sum(f['level'] for f in failures) / len(failures)
    print(f"  {strategy}: {len(failures)} failures (avg level: {avg_level:.3f})")
```

### Visualizing Failures

The exported failures include side-by-side images:

```
export_path/
├── manifest.json
├── 0_original.png       # Original image
├── 0_perturbed.png      # After perturbation
├── 1_original.png
├── 1_perturbed.png
└── ...
```

Open these in an image viewer to understand what causes failures.

---

## Advanced Configuration

### Custom Normalization

If your model uses specific normalization:

```python
# ImageNet normalization
report = quick_check(
    model,
    data,
    preset="standard",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# CIFAR-10 normalization
report = quick_check(
    model,
    data,
    preset="standard",
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010)
)
```

**Important:** Only pass `mean` and `std` if your test data is NOT already normalized. If you're using transforms with `Normalize()`, don't pass these parameters.

### Adjusting Budget

The budget controls precision vs speed:

```python
# Fast but less precise (development)
report = quick_check(model, data, preset="standard", budget=100)

# Normal (recommended)
report = quick_check(model, data, preset="standard", budget=1000)

# Slower but very precise (production validation)
report = quick_check(model, data, preset="standard", budget=5000)
```

**Rule of thumb:**
- Development: 100-500
- Production testing: 1000-2000
- Research/benchmarking: 5000+

### Device Selection

```python
# Auto-detect (default, recommended)
report = quick_check(model, data, preset="standard", device="auto")

# Force CUDA
report = quick_check(model, data, preset="standard", device="cuda")

# Force CPU (if GPU out of memory)
report = quick_check(model, data, preset="standard", device="cpu")

# Apple Silicon
report = quick_check(model, data, preset="standard", device="mps")
```

---

## Production Integration

### CI/CD Pipeline

```python
# test_robustness.py
from visprobe import quick_check
from your_project import load_model, get_test_data

def test_model_robustness():
    """CI test: ensure model meets robustness requirements."""
    model = load_model("checkpoints/latest.pth")
    test_data = get_test_data(num_samples=100)

    report = quick_check(
        model,
        test_data,
        preset="standard",
        budget=1000,
        device="cuda"
    )

    # Fail CI if robustness is too low
    assert report.score > 0.70, \
        f"Model robustness {report.score:.1%} below threshold 70%"

    # Save report for tracking
    report.save(f"reports/robustness_{os.environ['CI_COMMIT_SHA']}.json")

    return report

if __name__ == "__main__":
    test_model_robustness()
```

### Monitoring Dashboard

```python
import streamlit as st
from visprobe import quick_check

@st.cache_resource
def run_robustness_test():
    model = load_production_model()
    data = sample_production_traffic()
    return quick_check(model, data, preset="standard")

report = run_robustness_test()

st.metric("Robustness Score", f"{report.score:.1%}")
st.metric("Total Failures", len(report.failures))

if report.score < 0.70:
    st.error("⚠️ Model robustness below threshold!")
```

### Automated Retraining Trigger

```python
def monitor_and_retrain():
    while True:
        # Sample recent production data
        test_data = sample_recent_traffic(hours=24)

        # Test robustness
        report = quick_check(production_model, test_data, preset="standard")

        # Trigger retraining if score drops
        if report.score < ROBUSTNESS_THRESHOLD:
            export_path = report.export_failures(n=100)
            trigger_retraining_job(hard_examples=export_path)

        time.sleep(3600)  # Check hourly
```

---

## Best Practices

### 1. Start Small, Then Scale

```python
# Development: Quick iteration
report = quick_check(model, data[:10], preset="standard", budget=100)

# Production: Full validation
report = quick_check(model, data, preset="standard", budget=1000)
```

### 2. Match Normalization to Training

Always ensure test data normalization matches training:

```python
# If trained with ImageNet normalization
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Don't pass mean/std to quick_check (already normalized)
report = quick_check(model, data, preset="standard")
```

### 3. Test with Representative Data

Use test data that matches your production distribution:

```python
# Good: Real test set
test_data = load_test_set()

# Bad: Random noise
test_data = [(torch.randn(3, 224, 224), 0) for _ in range(100)]
```

### 4. Compare Across Presets

Find your model's weakest area:

```python
results = {}
for preset_name in ["standard", "lighting", "blur", "corruption"]:
    report = quick_check(model, data, preset=preset_name, budget=500)
    results[preset_name] = report.score

weakest = min(results.items(), key=lambda x: x[1])
print(f"Weakest area: {weakest[0]} (score: {weakest[1]:.1%})")
```

### 5. Track Over Time

```python
# Save each version's results
report = quick_check(model, data, preset="standard")
report.save(f"robustness_reports/v{model_version}.json")

# Compare with previous version
import json
with open(f"robustness_reports/v{model_version-1}.json") as f:
    prev_score = json.load(f)['metrics']['overall_robustness_score']

improvement = report.score - prev_score
print(f"Improvement: {improvement*100:+.1f}%")
```

### 6. Export and Retrain on Failures

```python
if report.score < 0.80:
    export_path = report.export_failures(n=50)

    # Add to next training run
    with open('hard_examples_paths.txt', 'a') as f:
        f.write(export_path + '\n')
```

---

## Common Workflows

### Workflow 1: First-Time Model Testing

```python
# 1. Quick sanity check
report = quick_check(model, data[:20], preset="standard", budget=100)
print(f"Quick score: {report.score:.1%}")

# 2. Full test if sanity check passes
if report.score > 0.50:
    full_report = quick_check(model, data, preset="standard", budget=1000)
    full_report.show()
    full_report.save("initial_robustness_report.json")
```

### Workflow 2: Compare Model Versions

```python
models = {
    'baseline': load_model('baseline.pth'),
    'v1': load_model('v1.pth'),
    'v2': load_model('v2.pth'),
}

for name, model in models.items():
    report = quick_check(model, test_data, preset="standard")
    print(f"{name}: {report.score:.1%}")
```

### Workflow 3: Find and Fix Weaknesses

```python
# 1. Test all presets
presets = ["standard", "lighting", "blur", "corruption"]
results = {p: quick_check(model, data, preset=p) for p in presets}

# 2. Find weakest
weakest = min(results.items(), key=lambda x: x[1].score)
print(f"Weakest: {weakest[0]} ({weakest[1].score:.1%})")

# 3. Export failures from weakest area
export_path = weakest[1].export_failures(n=100)

# 4. Retrain with hard examples
# ... your retraining code ...

# 5. Re-test
new_report = quick_check(retrained_model, data, preset=weakest[0])
print(f"Improvement: {(new_report.score - weakest[1].score)*100:.1f}%")
```

### Workflow 4: Production Validation

```python
def validate_before_deployment(checkpoint_path):
    model = load_checkpoint(checkpoint_path)
    test_data = load_production_test_set()

    report = quick_check(model, test_data, preset="standard", budget=2000)

    criteria = {
        'robustness_score': (report.score, 0.70),
        'max_failures': (len(report.failures), 30),
    }

    passed = all(value >= threshold for value, threshold in criteria.values())

    if passed:
        print("✅ Model passed all criteria - ready for deployment")
        deploy_model(checkpoint_path)
    else:
        print("❌ Model failed validation criteria:")
        for metric, (value, threshold) in criteria.items():
            status = "✅" if value >= threshold else "❌"
            print(f"  {status} {metric}: {value} (threshold: {threshold})")

        # Export failures for debugging
        report.export_failures(n=50)

    return passed
```

---

## Next Steps

- **See working examples**: Check out [examples on GitHub](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples)
- **Full API reference**: See [`COMPREHENSIVE_API_REFERENCE.md`](../COMPREHENSIVE_API_REFERENCE.md)
- **Troubleshooting**: See [`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md)
- **Report issues**: [GitHub Issues](https://github.com/bilgedemirkaya/VisProbe/issues)

---

**Happy testing!** 🚀
