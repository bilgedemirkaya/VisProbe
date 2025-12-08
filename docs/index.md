# VisProbe Documentation

**VisProbe** is a Python library for interactive robustness testing of machine learning models. It provides a declarative API for defining robustness properties and automatically searching for failure thresholds under various perturbations.

## Overview

VisProbe enables you to:

- **Test robustness properties** using simple decorators (@given, @search)
- **Find failure thresholds** automatically with multiple search algorithms
- **Visualize results** in an interactive dashboard
- **Analyze model behavior** across different perturbations, resolutions, and noise levels
- **Track intermediate layers** to understand internal representations

## Quick Example

```python
import torch
from visprobe import model, data_source, search
from visprobe.strategies.image import GaussianNoiseStrategy
from visprobe.properties.classification import LabelConstant

# Load your model and data
my_model = load_pretrained_model()
test_images = load_test_images()

@model(my_model)
@data_source(test_images, class_names=["cat", "dog"])
@search(
    strategy=lambda level: GaussianNoiseStrategy(std_dev=level),
    initial_level=0.001,
    mode='binary',
    level_lo=0.0,
    level_hi=0.5
)
def test_noise_robustness(original, perturbed):
    """Find the noise level where predictions start to fail."""
    return LabelConstant.evaluate(original, perturbed)
```

## Key Features

### 1. Declarative API
Define tests using Python decorators - no boilerplate code needed.

### 2. Multiple Search Modes
- **Adaptive**: Step-halving search for efficient threshold finding
- **Binary**: O(log n) binary search for precise thresholds
- **Grid**: Uniform sampling across the perturbation space
- **Random**: Random sampling for exploration

### 3. Rich Perturbation Strategies
- **Natural transformations**: Gaussian noise, brightness, contrast, rotation
- **Adversarial attacks**: FGSM, PGD, BIM, Auto-PGD, Square Attack
- **Composite strategies**: Chain multiple perturbations

### 4. Comprehensive Analysis
- Per-sample threshold estimation
- Resolution impact analysis
- Noise sensitivity sweeps
- Corruption robustness (CIFAR-10-C style)
- Intermediate layer analysis
- Top-k prediction overlap tracking

### 5. Interactive Visualization
Built-in Streamlit dashboard for exploring results with:
- Side-by-side image comparison
- Search path visualization
- Metrics and statistics
- Exportable reports

## Installation

```bash
pip install visprobe

# For adversarial attacks (optional)
pip install adversarial-robustness-toolbox

# For visualization (optional)
pip install streamlit
```

## Documentation Structure

- **[Architecture](architecture.md)** - System design and module organization
- **[User Guide](user-guide.md)** - Getting started and common workflows
- **[API Reference](api/index.md)** - Detailed API documentation
- **[Design Rationale](design-rationale.md)** - Why we made certain design choices
- **[Examples](examples/index.md)** - Complete code examples

## Quick Links

- [GitHub Repository](https://github.com/bilgedemirkaya/visprobe)
- [Report Issues](https://github.com/bilgedemirkaya/visprobe/issues)

## Citation

If you use VisProbe in your research, please cite:

```bibtex
@software{visprobe,
  title={VisProbe: Interactive Robustness Testing for ML Models},
  author={Bilge Demirkaya},
  year={2025},
  url={https://github.com/your-repo/visprobe}
}
```
