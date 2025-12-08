# API Reference

Complete API documentation for VisProbe.

## Decorators

### @model

```python
@model(model_obj: Any, *, capture_intermediate_layers: Optional[List[str]] = None)
```

Attaches a PyTorch model to a VisProbe test.

**Parameters:**
- `model_obj`: The PyTorch model to test
- `capture_intermediate_layers`: Optional list of layer names to capture during forward pass

**Example:**
```python
@model(my_resnet, capture_intermediate_layers=["layer4"])
@data_source(test_data)
@given(strategy=GaussianNoiseStrategy(std=0.05))
def test_robustness(original, perturbed):
    # original["features"] will contain layer4 outputs
    return LabelConstant.evaluate(original, perturbed)
```

---

### @data_source

```python
@data_source(
    data_obj: Any,
    *,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    class_names: Optional[List[str]] = None,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
)
```

Provides the data source for a VisProbe test.

**Parameters:**
- `data_obj`: Data source (tensor, dataset, or any object)
- `collate_fn`: Optional function to collate data into batches
- `class_names`: Optional list of class names for visualization
- `mean`: Channel means for denormalization (defaults to ImageNet means)
- `std`: Channel stds for denormalization (defaults to ImageNet stds)

**Example:**
```python
@data_source(
    test_images,
    class_names=["cat", "dog", "bird"],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

---

### @given

```python
@given(
    *,
    strategy: Strategy,
    vectorized: bool = False,
    noise_sweep: Optional[Dict[str, Any]] = DEFAULT_NOISE_SWEEP,
    resolutions: Optional[List[Tuple[int, int]]] = DEFAULT_RESOLUTIONS,
    top_k: Optional[int] = 5,
    property_name: Optional[str] = None,
)
```

Defines a test with a fixed perturbation level.

**Parameters:**
- `strategy`: Perturbation strategy instance
- `vectorized`: If True, evaluate property on full batch (faster)
- `noise_sweep`: Parameters for noise sensitivity analysis
- `resolutions`: List of (height, width) tuples to test
- `top_k`: Number of top predictions to analyze
- `property_name`: Display name for the property

**Returns:** Test function that returns a `Report` object

**Example:**
```python
@given(
    strategy=GaussianNoiseStrategy(std_dev=0.1),
    vectorized=True,
    property_name="Label Stability under Noise"
)
def test_noise(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

---

### @search

```python
@search(
    *,
    strategy: Callable[[float], Strategy] | Strategy,
    initial_level: float = 0.001,
    step: float = 0.002,
    min_step: float = 1e-5,
    max_queries: int = 500,
    mode: str = "adaptive",
    level_lo: Optional[float] = None,
    level_hi: Optional[float] = None,
    resolutions: Optional[List[Tuple[int, int]]] = DEFAULT_RESOLUTIONS,
    noise_sweep: Optional[Dict[str, Any]] = DEFAULT_NOISE_SWEEP,
    top_k: Optional[int] = 5,
    reduce: Optional[str] = "all",
    property_name: Optional[str] = None,
)
```

Defines a search for a model's failure point.

**Parameters:**
- `strategy`: Perturbation strategy or factory function `lambda level: Strategy(level)`
- `initial_level`: Starting level for search
- `step`: Step size for adaptive search
- `min_step`: Minimum step size before stopping
- `max_queries`: Maximum model queries allowed
- `mode`: Search mode - `'adaptive'`, `'binary'`, `'grid'`, or `'random'`
- `level_lo`: Lower bound for search (used by binary/grid/random)
- `level_hi`: Upper bound for search (used by binary/grid/random)
- `resolutions`: Resolutions to test for analysis
- `noise_sweep`: Parameters for noise sensitivity analysis
- `top_k`: Number of top predictions to analyze
- `reduce`: Aggregation method (`'all'`, `'any'`, or `'frac>=X'`)
- `property_name`: Display name for the robustness property

**Returns:** Test function that returns a `Report` object

**Example:**
```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode='binary',
    level_lo=0.0,
    level_hi=0.5,
    reduce="frac>=0.8"
)
def test_threshold(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

## Strategies

### Base Strategy

```python
class Strategy:
    def generate(self, imgs: Any, model: Any, level: Optional[float] = None) -> Any:
        """Generate perturbed version of imgs."""

    def query_cost(self) -> int:
        """Return number of extra model queries required."""
```

### Image Strategies

#### GaussianNoiseStrategy

```python
class GaussianNoiseStrategy(Strategy):
    def __init__(
        self,
        std_dev: float,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        seed: Optional[int] = None,
    )
```

Additive Gaussian noise in pixel space.

#### BrightnessStrategy

```python
class BrightnessStrategy(Strategy):
    def __init__(self, brightness_factor: float)
```

Brightness adjustment (factor > 1.0 brightens, < 1.0 darkens).

#### ContrastStrategy

```python
class ContrastStrategy(Strategy):
    def __init__(self, contrast_factor: float)
```

Contrast adjustment (factor > 1.0 increases, < 1.0 decreases).

#### RotateStrategy

```python
class RotateStrategy(Strategy):
    def __init__(self, angle: float)
```

Rotation by angle in degrees (counter-clockwise).

### Adversarial Strategies

Require `pip install adversarial-robustness-toolbox`

#### FGSMStrategy

```python
class FGSMStrategy(Strategy):
    def __init__(
        self,
        eps: float = 2/255,
        targeted: bool = False,
        art_attack_kwargs: dict = None
    )
```

Fast Gradient Sign Method attack.

#### PGDStrategy

```python
class PGDStrategy(Strategy):
    def __init__(
        self,
        eps: float,
        eps_step: Optional[float] = None,
        max_iter: int = 100,
        **kwargs
    )
```

Projected Gradient Descent attack.

#### BIMStrategy

```python
class BIMStrategy(Strategy):
    def __init__(
        self,
        eps: float,
        eps_step: Optional[float] = None,
        max_iter: int = 10,
        **kwargs
    )
```

Basic Iterative Method (I-FGSM) attack.

#### APGDStrategy

```python
class APGDStrategy(Strategy):
    def __init__(self, eps: float, max_iter: int = 100, **kwargs)
```

Auto-PGD with adaptive step sizes.

#### SquareAttackStrategy

```python
class SquareAttackStrategy(Strategy):
    def __init__(self, eps: float, max_iter: int = 5000, **kwargs)
```

Score-based black-box attack.

## Properties

### Base Property

```python
class Property:
    def __call__(self, original: Any, perturbed: Any) -> bool:
        """Check if property holds."""

    @classmethod
    def evaluate(cls, original: Any, perturbed: Any, **init_kwargs) -> bool:
        """Convenience method for one-time evaluation."""
```

### Classification Properties

#### LabelConstant

```python
class LabelConstant(Property):
    """Top-1 label must remain constant."""
```

**Example:**
```python
LabelConstant.evaluate(original, perturbed)
```

#### TopKStability

```python
class TopKStability(Property):
    def __init__(
        self,
        k: int = 5,
        mode: str = "overlap",
        *,
        min_overlap: int = 3,
        require_containment: bool = True,
        min_jaccard: float = 0.4,
    )
```

Top-k predictions must satisfy stability criterion.

**Modes:**
- `"overlap"`: At least `min_overlap` common classes
- `"containment"`: Original top-1 in perturbed top-k
- `"jaccard"`: Jaccard index >= `min_jaccard`

**Example:**
```python
TopKStability.evaluate(
    original, perturbed,
    k=5,
    mode="overlap",
    min_overlap=3
)
```

#### ConfidenceDrop

```python
class ConfidenceDrop(Property):
    def __init__(self, max_drop: float = 0.3)
```

Confidence must not drop by more than `max_drop`.

#### L2Distance

```python
class L2Distance(Property):
    def __init__(self, max_delta: float = 1.0)
```

L2 distance between output logits.

## Report

### Report Dataclass

```python
@dataclass
class Report:
    test_name: str
    test_type: str  # "given" or "search"
    runtime: float
    model_queries: int

    # Optional fields
    model_name: Optional[str]
    property_name: Optional[str]
    strategy: Optional[str]
    failure_threshold: Optional[float]
    total_samples: Optional[int]
    passed_samples: Optional[int]
    # ... many more fields

    @property
    def robust_accuracy(self) -> Optional[float]:
        """For @given tests: passed_samples / total_samples"""

    def to_json(self) -> str:
        """Serialize to JSON string"""

    def save(self):
        """Save to file system"""
```

### ImageData

```python
@dataclass
class ImageData:
    image_b64: str
    prediction: str
    confidence: float

    @classmethod
    def from_tensors(
        cls,
        tensor: torch.Tensor,
        output: torch.Tensor,
        class_names: Optional[List[str]],
        mean: Optional[List[float]],
        std: Optional[List[float]],
    ) -> "ImageData"
```

## Utility Functions

### cifar10_data_source

```python
def cifar10_data_source(
    dataset: Any,
    *,
    normalized: bool = False,
    meta_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[Any, Callable, Optional[List[str]], List[float], List[float]]
```

Convenience helper for CIFAR-10 datasets.

**Returns:** `(dataset, collate_fn, class_names, mean, std)`

**Example:**
```python
from torchvision.datasets import CIFAR10
from visprobe.api.utils import cifar10_data_source

dataset = CIFAR10(root='./data', train=False, download=True)
data, collate, classes, mean, std = cifar10_data_source(
    dataset,
    normalized=True  # If using Normalize transform
)

@data_source(data, collate_fn=collate, class_names=classes, mean=mean, std=std)
```

### to_image_space

```python
def to_image_space(imgs: torch.Tensor, mean, std) -> torch.Tensor
```

Convert normalized tensors to image space [0,1].

### to_model_space

```python
def to_model_space(imgs: torch.Tensor, mean, std) -> torch.Tensor
```

Normalize image-space tensors for model input.

## Configuration

### Environment Variables

- `VISPROBE_DEVICE`: Device to use (`cpu`, `cuda`, `mps`, `hip`)
- `VISPROBE_PREFER_GPU`: If `1`, auto-select best GPU
- `VISPROBE_SEED`: Random seed for reproducibility
- `VISPROBE_RESULTS_DIR`: Directory for saving reports
- `VISPROBE_MODULE_NAME`: Override module name in reports
- `VISPROBE_DEBUG`: Enable debug logging

### Device Selection

```python
from visprobe.api.config import get_default_device

device = get_default_device()  # Smart default selection
```

Priority:
1. `VISPROBE_DEVICE` environment variable
2. Auto-detect with `VISPROBE_PREFER_GPU=1`
3. Default to CPU

## CLI

### Commands

```bash
# Run tests
visprobe run test_file.py [--device DEVICE] [--keep]

# Visualize results
visprobe visualize test_file.py [--device DEVICE]
```

**Options:**
- `--device`: Device to use (`cpu`, `cuda`, `mps`, `hip`, `auto`)
- `--keep`: Keep previous JSON result files (for `run` command)

## Type Hints

VisProbe uses type hints throughout. For best experience:

```bash
# Install type stubs
pip install types-torch types-torchvision

# Run type checker
mypy your_test.py
```

## Thread Safety

- **TestRegistry**: Thread-local storage, safe for parallel execution
- **QueryCounter**: Instance-specific hooks, no global state
- **Strategy caching**: Per-instance caches keyed by device/model

Safe to run tests in parallel:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(test_fn) for test_fn in tests]
    reports = [f.result() for f in futures]
```
