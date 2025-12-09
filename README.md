# VisProbe

**Find robustness failures in your vision models in 5 minutes.**

VisProbe automatically tests your model against natural perturbations (blur, lighting, compression) and compositional edge cases that standard tests miss. Get a robustness score, failure cases, and actionable insights—with just 3 lines of code.

```python
from visprobe import quick_check
import torchvision.models as models

model = models.resnet50(weights='DEFAULT')
report = quick_check(model, your_data, preset="standard")
report.show()  # → Robustness score: 67.3%, 12 failures found
```

**Why VisProbe?**
- ✅ **5-minute setup** - No boilerplate, no config files
- ✅ **Finds real failures** - Compositional perturbations (low-light + blur, compression + noise)
- ✅ **Actionable results** - Export worst cases as training data
- ✅ **Production-ready** - Validated presets, clear documentation

---

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from visprobe import quick_check
import torchvision.models as models
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

# 1. Load your model
model = models.resnet18(weights='DEFAULT')
model.eval()

# 2. Prepare test data (any format works: DataLoader, list, tensors)
transform = T.Compose([T.Resize(224), T.ToTensor()])
dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_data = [dataset[i] for i in range(100)]  # Test on 100 samples

# 3. Run robustness test
report = quick_check(model, test_data, preset="standard")

# 4. View results
report.show()
# ============================================================
# VisProbe Report: quick_check_standard
# ============================================================
# Robustness score: 67.3%
# Failures found: 12
# Total samples: 100
#
# Per-Strategy Results:
# ------------------------------------------------------------
#   brightness                     Score: 85.2%  Threshold: 1.32
#   gaussian_blur                  Score: 72.8%  Threshold: 2.10
#   gaussian_noise                 Score: 81.5%  Threshold: 0.027
#   jpeg_compression               Score: 90.1%  Threshold: 45
#   low_light_blur (compositional) Score: 38.2%  Threshold: 0.52
#   compressed_noisy (compositional) Score: 42.7%  Threshold: 28
# ============================================================

# 5. Export failures for retraining
report.export_failures(n=10)
# ✅ Exported 10 failures to visprobe_results/failures/quick_check_standard
```

That's it! You now have:
- **Robustness score** (0-1, higher is better)
- **Failure cases** with original vs perturbed predictions
- **Per-perturbation thresholds** showing where your model breaks
- **Exported failure dataset** ready for retraining

---

## 📊 What You Get

### 1. Robustness Score
A single number (0-1) summarizing your model's robustness across multiple perturbations.

### 2. Failure Cases
Detailed information about each failure:
- Original and perturbed predictions
- Perturbation type and severity
- Confidence drops

### 3. Actionable Insights
```python
# Get summary dict
print(report.summary)
# {'score': 0.673, 'total_failures': 12, 'runtime_sec': 45.2, ...}

# Access failures programmatically
for failure in report.failures[:5]:
    print(f"Sample {failure['index']}: {failure['original_pred']} → {failure['perturbed_pred']}")

# Export worst cases for retraining
report.export_failures(n=20, output_dir="./hard_cases")
```

---

## 🎨 Presets

VisProbe includes 4 validated presets covering common robustness concerns:

| Preset | Use Case | Perturbations | Time (100 imgs) |
|--------|----------|---------------|-----------------|
| **`standard`** | General robustness, pre-deployment | Brightness, blur, noise, compression, **+ compositional** | ~15 min |
| **`lighting`** | Outdoor cameras, varying daylight | Brightness, contrast, gamma, **+ low-light scenarios** | ~8 min |
| **`blur`** | Motion, defocus, video frames | Gaussian blur, motion blur, compression | ~10 min |
| **`corruption`** | Lossy transmission, noisy sensors | Noise, compression, degradation | ~10 min |

### What Makes Presets Special?

**Validated Ranges**: Each perturbation range is manually validated to preserve label semantics (~85-90% of perturbed images are still recognizable).

**Compositional Perturbations**: Standard tests miss failures that only occur with *multiple* perturbations applied together. VisProbe tests:
- **Low-light + blur** (e.g., nighttime dash cam)
- **Compression + noise** (e.g., degraded video transmission)
- **Dim + low contrast** (e.g., indoor cameras)

This finds **different failures** than testing perturbations individually.

### Example: Choosing a Preset

```python
# Testing an outdoor security camera model?
report = quick_check(model, data, preset="lighting")

# Testing a video processing model?
report = quick_check(model, data, preset="blur")

# General pre-deployment check?
report = quick_check(model, data, preset="standard")
```

---

## 💡 Why VisProbe vs. Standard Testing?

| Approach | Coverage | Setup Time | Compositional | Actionable |
|----------|----------|------------|---------------|------------|
| **Random test images** | ❌ Low | 5 min | ❌ No | ❌ No |
| **ImageNet-C** | ⚠️ Fixed corruptions | 30 min | ❌ No | ⚠️ Limited |
| **Manual adversarial** | ⚠️ Targeted | Hours | ❌ No | ✅ Yes |
| **VisProbe** | ✅ Adaptive | **5 min** | ✅ Yes | ✅ Yes |

**VisProbe advantages:**
1. **Adaptive search** finds the *exact threshold* where your model fails (not just pass/fail)
2. **Compositional perturbations** catch edge cases that single perturbations miss
3. **Export failures** directly as training data for improvement
4. **5-minute setup** with sensible defaults

---

## 🔧 Advanced Usage

### Custom Test Budget

```python
# Quick test (fewer model queries)
report = quick_check(model, data, preset="standard", budget=500)

# Thorough test (more precise thresholds)
report = quick_check(model, data, preset="standard", budget=5000)
```

### Device Selection

```python
# Auto-detect best device (default)
report = quick_check(model, data, preset="standard", device="auto")

# Force specific device
report = quick_check(model, data, preset="standard", device="cuda")
report = quick_check(model, data, preset="standard", device="cpu")
```

### Custom Normalization

```python
# CIFAR-10 normalization
report = quick_check(
    model, data, preset="standard",
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616)
)
```

### Jupyter Integration

```python
# In a Jupyter notebook, report.show() displays rich HTML:
report.show()
```

![Jupyter Report Example](docs/images/jupyter_report.png)

---

## 📚 Examples

See the `examples/` directory for more:

- **[quickstart.ipynb](examples/quickstart.ipynb)** - Complete walkthrough with CIFAR-10
- **[cifar10_test.py](examples/cifar10_test.py)** - Testing ResNet on CIFAR-10
- **[custom_model.py](examples/custom_model.py)** - Using your own model
- **[comparison.py](examples/comparison.py)** - Comparing multiple models

---

## 🏗️ Advanced API (Power Users)

For researchers who need more control, VisProbe provides a decorator-based API:

```python
from visprobe import given, search, model, data_source
from visprobe.strategies import GaussianNoiseStrategy
from visprobe.properties import LabelConstant

@search(
    strategy=lambda level: GaussianNoiseStrategy(std_dev=level),
    initial_level=0.0,
    step=0.01,
    max_queries=1000
)
@model(my_model)
@data_source(my_data)
def test_noise_robustness(original, perturbed):
    assert LabelConstant()(original, perturbed)

# Run with CLI
# $ visprobe run test_file.py
# $ visprobe visualize test_file.py
```

See [COMPREHENSIVE_API_REFERENCE.md](COMPREHENSIVE_API_REFERENCE.md) for full documentation.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we'd love help with:**
- Additional presets (weather, medical imaging, etc.)
- Support for object detection / segmentation
- Performance optimizations
- More examples and tutorials

---

## 📖 Citation

If you use VisProbe in your research, please cite:

```bibtex
@software{visprobe2024,
  title={VisProbe: Adaptive Robustness Testing for Vision Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/VisProbe}
}
```

---

## 📄 License

VisProbe is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **API Reference**: [COMPREHENSIVE_API_REFERENCE.md](COMPREHENSIVE_API_REFERENCE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Performance Guide**: [PERFORMANCE.md](PERFORMANCE.md)

---

## ⭐ Show Your Support

If VisProbe helped you find robustness issues, give us a star! ⭐

**Questions?** Open an issue or start a discussion.

**Found a bug?** We appreciate bug reports with minimal reproducible examples.

---
