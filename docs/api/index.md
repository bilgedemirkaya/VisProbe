# API Reference

Quick reference for the main VisProbe functions and classes.

## Main Functions

### `quick_check()`

The main entry point for robustness testing.

```python
from visprobe import quick_check

report = quick_check(
    model=your_model,              # PyTorch model
    data=test_data,                # List of (image, label) tuples or DataLoader
    preset="natural",              # "natural", "adversarial", "realistic_attack", "comprehensive"
    budget=1000,                   # Model queries per strategy (higher = more precise)
    device="auto",                 # "auto", "cuda", "cpu", "mps"
    mean=None,                     # Normalization mean (optional)
    std=None                       # Normalization std (optional)
)
```

**Returns:** `Report` object with results

### `compare_threat_models()`

Compare all three threat models at once.

```python
from visprobe import compare_threat_models

results = compare_threat_models(
    model=your_model,
    data=test_data,
    budget=1000
)

print(results['scores'])  # {'natural': 0.75, 'adversarial': 0.65, 'realistic_attack': 0.55}
```

## Report Object

Access and export results.

```python
# Properties
report.score                    # Overall robustness (0-1)
report.threat_model             # Which threat model was used
report.threat_model_scores      # Per-threat-model scores
report.vulnerability_warning    # Warning if vulnerable to opportunistic attacks
report.failures                 # List of failures
report.summary                  # Dictionary of metrics

# Methods
report.show()                   # Rich interactive display
report.export_failures(n=10)    # Export top N failures
report.save(path)               # Save to JSON
```

## Available Presets

```
"natural"          - Environmental perturbations
"adversarial"      - Gradient-based attacks (requires ART)
"realistic_attack" - Attacks in real conditions (requires ART)
"comprehensive"    - All three threat models
```

Legacy presets still supported: `"standard"`, `"lighting"`, `"blur"`, `"corruption"`

## Strategies

Import from `visprobe.strategies.image`:

```python
from visprobe.strategies.image import (
    BrightnessStrategy,
    ContrastStrategy,
    GammaStrategy,
    GaussianBlurStrategy,
    MotionBlurStrategy,
    JPEGCompressionStrategy,
    GaussianNoiseStrategy
)
```

## Properties

Import from `visprobe.properties.classification`:

```python
from visprobe.properties.classification import (
    LabelConstant,           # Top-1 prediction must stay same
    TopKStability,           # Top-k predictions stable
    ConfidenceDrop,          # Limit confidence decrease
    L2Distance               # L2 distance between logits
)
```

## Complete Documentation

For full parameters and advanced usage: See [COMPREHENSIVE_API_REFERENCE.md](../COMPREHENSIVE_API_REFERENCE.md)
