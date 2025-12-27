# API Reference

**For the complete API reference, see:** [COMPREHENSIVE_API_REFERENCE.md](../COMPREHENSIVE_API_REFERENCE.md)

## Quick Reference

### Main Functions

```python
from visprobe import quick_check, compare_threat_models

# Single threat model test
report = quick_check(
    model=your_model,
    data=test_data,
    preset="natural",  # Threat-model-aware preset
    budget=1000,
    device="auto"
)

# Compare all threat models
results = compare_threat_models(your_model, test_data, budget=1000)
```

### Report Object

```python
# Access results
score = report.score                           # Overall robustness (0-1)
threat_model = report.threat_model             # The threat model used
threat_model_scores = report.threat_model_scores  # Per-threat-model breakdown
vulnerability_warning = report.vulnerability_warning  # Opportunistic attack warning
failures = report.failures                     # List of failure cases
summary = report.summary                       # Metrics dictionary

# Display results
report.show()                                  # Context-aware display

# Export failures
path = report.export_failures(n=10)            # Export top 10
```

### Available Presets (Threat-Model-Aware)

**Passive Threat Model (No Adversary):**
- **`"natural"`** - Environmental perturbations (brightness, blur, noise, compression)

**Active Threat Model (Adversarial):**
- **`"adversarial"`** - Gradient-based attacks (FGSM, PGD, BIM)

**Active + Environmental (Realistic):**
- **`"realistic_attack"`** ⭐ - Attacks under suboptimal conditions (low-light + FGSM, etc.)

**All Threat Models:**
- **`"comprehensive"`** - Complete evaluation with per-threat-model breakdown

**Legacy Presets (Deprecated):**
- **`"standard"`**, **`"lighting"`**, **`"blur"`**, **`"corruption"`** - Still supported with deprecation warnings

### Strategies

All strategies from `visprobe.strategies.image`:

- `BrightnessStrategy(brightness_factor)`
- `ContrastStrategy(contrast_factor)`
- `GammaStrategy(gamma, gain=1.0)`
- `GaussianBlurStrategy(kernel_size, sigma)`
- `MotionBlurStrategy(kernel_size, angle)`
- `JPEGCompressionStrategy(quality)`
- `GaussianNoiseStrategy(std_dev, mean=None, std=None)`

### Properties

From `visprobe.properties.classification`:

- `LabelConstant()` - Top-1 prediction must stay the same
- `TopKStability(k, mode, ...)` - Top-k predictions stability
- `ConfidenceDrop(max_drop)` - Limit confidence decrease
- `L2Distance(max_delta)` - L2 distance between logits

## Complete Documentation

For full API documentation with all parameters, examples, and advanced usage:

**→ See [COMPREHENSIVE_API_REFERENCE.md](../COMPREHENSIVE_API_REFERENCE.md)**

## Examples

For working code examples, visit the [examples/ folder on GitHub](https://github.com/bilgedemirkaya/VisProbe/tree/main/examples):

- `basic_example.py` - Minimal 3-line usage
- `cifar10_example.py` - Complete CIFAR-10 workflow
- `custom_model_example.py` - Template for your models
- `threat_model_comparison.py` - Compare all threat models
