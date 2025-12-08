# Examples

Complete code examples for using VisProbe.

## Basic Examples

### 1. Testing with Fixed Noise Level

```python
import torch
from torchvision import models
from visprobe import model, data_source, given
from visprobe.strategies.image import GaussianNoiseStrategy
from visprobe.properties.classification import LabelConstant

# Load model
resnet = models.resnet18(pretrained=True)
resnet.eval()

# Test data
test_images = torch.randn(16, 3, 224, 224)

@model(resnet)
@data_source(test_images)
@given(strategy=GaussianNoiseStrategy(std_dev=0.05))
def test_noise_robustness(original, perturbed):
    """Test if predictions remain constant under σ=0.05 Gaussian noise."""
    return LabelConstant.evaluate(original, perturbed)

if __name__ == "__main__":
    report = test_noise_robustness()
    print(f"Robust accuracy: {report.robust_accuracy:.2%}")
    print(f"Passed: {report.passed_samples}/{report.total_samples}")
```

### 2. Finding Failure Threshold

```python
from visprobe import search

@model(resnet)
@data_source(test_images)
@search(
    strategy=lambda level: GaussianNoiseStrategy(std_dev=level),
    mode='binary',
    level_lo=0.0,
    level_hi=0.5,
    max_queries=100
)
def test_noise_threshold(original, perturbed):
    """Find the noise level where model starts failing."""
    return LabelConstant.evaluate(original, perturbed)

if __name__ == "__main__":
    report = test_noise_threshold()
    print(f"Failure threshold: {report.failure_threshold:.4f}")
    print(f"Queries used: {report.model_queries}")
```

## Advanced Examples

### 3. CIFAR-10 with Custom Normalization

```python
from torchvision import datasets, transforms
from visprobe.api.utils import cifar10_data_source

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2470, 0.2435, 0.2616])
])

dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Take first 32 samples
indices = list(range(32))
subset = torch.utils.data.Subset(dataset, indices)

# Setup with correct normalization
data, collate, classes, mean, std = cifar10_data_source(
    subset,
    normalized=True
)

@model(my_cifar_model)
@data_source(data, collate_fn=collate, class_names=classes, mean=mean, std=std)
@search(strategy=lambda l: GaussianNoiseStrategy(std_dev=l))
def test_cifar_robustness(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

### 4. Adversarial Attack Testing

```python
from visprobe.strategies.adversarial import FGSMStrategy, PGDStrategy

@model(resnet)
@data_source(test_images)
@search(
    strategy=lambda eps: FGSMStrategy(eps=eps),
    mode='binary',
    level_lo=0.0,
    level_hi=0.1,
    property_name="FGSM Robustness"
)
def test_fgsm_threshold(original, perturbed):
    """Find FGSM epsilon threshold."""
    return LabelConstant.evaluate(original, perturbed)

@model(resnet)
@data_source(test_images)
@search(
    strategy=lambda eps: PGDStrategy(eps=eps, max_iter=40),
    mode='adaptive',
    initial_level=0.001,
    property_name="PGD Robustness"
)
def test_pgd_threshold(original, perturbed):
    """Find PGD epsilon threshold."""
    return LabelConstant.evaluate(original, perturbed)
```

### 5. Top-K Stability Testing

```python
from visprobe.properties.classification import TopKStability

@model(resnet)
@data_source(test_images)
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='binary',
    level_lo=0.0,
    level_hi=0.3,
    property_name="Top-5 Overlap"
)
def test_topk_stability(original, perturbed):
    """Test if top-5 predictions have at least 3 overlapping classes."""
    return TopKStability.evaluate(
        original, perturbed,
        k=5,
        mode="overlap",
        min_overlap=3
    )
```

### 6. Composite Perturbations

```python
from visprobe.strategies.image import BrightnessStrategy

# Chain multiple perturbations
composite_strategy = [
    GaussianNoiseStrategy(std_dev=0.02),
    BrightnessStrategy(brightness_factor=1.1)
]

@model(resnet)
@data_source(test_images)
@given(strategy=composite_strategy)
def test_combined_perturbations(original, perturbed):
    """Test robustness to noise + brightness."""
    return LabelConstant.evaluate(original, perturbed)
```

### 7. Capturing Intermediate Layers

```python
@model(resnet, capture_intermediate_layers=["layer4", "avgpool"])
@data_source(test_images)
@given(strategy=GaussianNoiseStrategy(std_dev=0.05))
def test_with_layer_analysis(original, perturbed):
    """Test with layer-wise similarity analysis."""
    # Report will include ensemble_analysis with layer cosine similarities
    return LabelConstant.evaluate(original, perturbed)

if __name__ == "__main__":
    report = test_with_layer_analysis()
    print("Layer-wise similarities:")
    for layer, similarity in report.ensemble_analysis.items():
        print(f"  {layer}: {similarity:.4f}")
```

### 8. Batch Reduction Strategies

```python
@model(resnet)
@data_source(test_images)
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='binary',
    level_lo=0.0,
    level_hi=0.3,
    reduce="frac>=0.8"  # 80% of samples must pass
)
def test_percentile_robustness(original, perturbed):
    """Find threshold where 80% of samples remain robust."""
    return LabelConstant.evaluate(original, perturbed)
```

### 9. Custom Property

```python
from visprobe.properties.base import Property
import torch

class MarginPreservation(Property):
    """Ensure prediction margin doesn't drop too much."""

    def __init__(self, min_margin=0.1):
        self.min_margin = min_margin

    def __call__(self, original, perturbed):
        orig_logits = original["output"]
        pert_logits = perturbed["output"]

        # Compute margins (difference between top-2 probabilities)
        orig_probs = torch.softmax(orig_logits, dim=-1)
        pert_probs = torch.softmax(pert_logits, dim=-1)

        orig_top2, _ = torch.topk(orig_probs, 2, dim=-1)
        pert_top2, _ = torch.topk(pert_probs, 2, dim=-1)

        pert_margin = pert_top2[:, 0] - pert_top2[:, 1]

        # All samples must maintain minimum margin
        return torch.all(pert_margin >= self.min_margin).item()

@model(resnet)
@data_source(test_images)
@search(strategy=lambda l: GaussianNoiseStrategy(std_dev=l))
def test_margin_preservation(original, perturbed):
    """Test if prediction margin is maintained."""
    return MarginPreservation(min_margin=0.15)(original, perturbed)
```

### 10. Custom Strategy

```python
from visprobe.strategies.base import Strategy
import torch.nn.functional as F

class SaltPepperNoise(Strategy):
    """Add salt-and-pepper noise to images."""

    def __init__(self, density=0.05):
        self.density = density

    def generate(self, imgs, model, level=None):
        density = level if level is not None else self.density
        noise_mask = torch.rand_like(imgs)

        imgs_noisy = imgs.clone()
        imgs_noisy[noise_mask < density/2] = 0  # Pepper (black)
        imgs_noisy[noise_mask > 1 - density/2] = 1  # Salt (white)

        return imgs_noisy

@model(resnet)
@data_source(test_images)
@search(strategy=lambda l: SaltPepperNoise(density=l), level_hi=0.2)
def test_salt_pepper(original, perturbed):
    """Test robustness to salt-and-pepper noise."""
    return LabelConstant.evaluate(original, perturbed)
```

## Running Examples

### Command Line

```bash
# Run a single test
python my_test.py

# Run and visualize
visprobe visualize my_test.py

# Run on GPU
visprobe run my_test.py --device cuda

# Set random seed
VISPROBE_SEED=42 python my_test.py
```

### Programmatic

```python
if __name__ == "__main__":
    # Run multiple tests
    tests = [
        test_noise_robustness,
        test_fgsm_threshold,
        test_topk_stability
    ]

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        report = test()
        print(f"  Result: {report.robust_accuracy or report.failure_threshold}")

    # Or in parallel
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(test) for test in tests]
        reports = [f.result() for f in futures]
```

## Complete Example: Autonomous Driving Scenario

```python
import torch
from torchvision import models
from visprobe import model, data_source, search
from visprobe.strategies.image import GaussianNoiseStrategy, BrightnessStrategy
from visprobe.properties.classification import LabelConstant, ConfidenceDrop

# Simulated autonomous driving model
driving_model = models.resnet50(pretrained=True)
driving_model.eval()

# Test scenarios
weather_scenarios = torch.randn(32, 3, 224, 224)  # Different weather conditions

@model(driving_model)
@data_source(
    weather_scenarios,
    class_names=["clear_road", "obstacle", "pedestrian", "vehicle"],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='binary',
    level_lo=0.0,
    level_hi=0.2,
    reduce="all",  # All scenarios must pass (safety-critical)
    property_name="Sensor Noise Robustness"
)
def test_sensor_noise(original, perturbed):
    """Test robustness to sensor noise."""
    return LabelConstant.evaluate(original, perturbed)

@model(driving_model)
@data_source(weather_scenarios)
@search(
    strategy=lambda l: BrightnessStrategy(brightness_factor=1.0 + l),
    mode='binary',
    level_lo=0.0,
    level_hi=0.5,
    property_name="Lighting Variation Robustness"
)
def test_lighting_changes(original, perturbed):
    """Test robustness to lighting changes."""
    # Allow small confidence drop but maintain label
    label_ok = LabelConstant.evaluate(original, perturbed)
    conf_ok = ConfidenceDrop.evaluate(original, perturbed, max_drop=0.2)
    return label_ok and conf_ok

if __name__ == "__main__":
    print("Testing autonomous driving model robustness...\n")

    sensor_report = test_sensor_noise()
    print(f"Sensor Noise Threshold: {sensor_report.failure_threshold:.4f}")
    print(f"  Queries used: {sensor_report.model_queries}")

    lighting_report = test_lighting_changes()
    print(f"\nLighting Variation Threshold: {lighting_report.failure_threshold:.4f}")
    print(f"  Queries used: {lighting_report.model_queries}")

    # Visualize results
    print("\nLaunch dashboard with: visprobe visualize your_test.py")
```

## Best Practices Examples

See the [User Guide](../user-guide.md#best-practices) for detailed best practices.
