# API Reference

**For the complete API reference, see:** [COMPREHENSIVE_API_REFERENCE.md](../../COMPREHENSIVE_API_REFERENCE.md)

## Quick Reference

### Main Function

```python
from visprobe import quick_check

report = quick_check(
    model=your_model,
    data=test_data,
    preset="standard",  # or "lighting", "blur", "corruption"
    budget=1000,
    device="auto"
)
```

### Report Object

```python
# Access results
score = report.score               # Overall robustness (0-1)
failures = report.failures         # List of failure cases
summary = report.summary           # Metrics dictionary

# Display results
report.show()                      # Context-aware display

# Export failures
path = report.export_failures(n=10)  # Export top 10
```

### Available Presets

- **`"standard"`** - General-purpose with compositional perturbations
- **`"lighting"`** - Brightness, contrast, gamma variations
- **`"blur"`** - Gaussian blur, motion blur, JPEG compression
- **`"corruption"`** - Noise and degradation

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

**→ See [COMPREHENSIVE_API_REFERENCE.md](../../COMPREHENSIVE_API_REFERENCE.md)**

## Examples

For working code examples:

**→ See [examples/](../../examples/)**

- `basic_example.py` - Minimal 3-line usage
- `cifar10_example.py` - Complete CIFAR-10 workflow
- `custom_model_example.py` - Template for your models
- `preset_comparison.py` - Compare all presets
