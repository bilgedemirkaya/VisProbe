# VisProbe Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using VisProbe.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Device and Memory Issues](#device-and-memory-issues)
- [Import and Module Errors](#import-and-module-errors)
- [Data Format Issues](#data-format-issues)
- [Normalization Issues](#normalization-issues)
- [Performance Issues](#performance-issues)
- [Result Interpretation](#result-interpretation)

---

## Installation Issues

### Problem: `pip install` fails with dependency conflicts

**Symptoms:**
```
ERROR: Cannot install visprobe because these package versions have conflicting dependencies
```

**Solutions:**

1. Create a fresh virtual environment:
   ```bash
   python -m venv visprobe_env
   source visprobe_env/bin/activate  # On Windows: visprobe_env\Scripts\activate
   pip install -e .
   ```

2. Install with specific PyTorch version:
   ```bash
   # For CPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -e .

   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install -e .
   ```

3. Install minimal dependencies first:
   ```bash
   pip install torch torchvision numpy pillow
   pip install -e .
   ```

### Problem: `ModuleNotFoundError: No module named 'visprobe'`

**Solutions:**

1. Ensure you're in the correct directory:
   ```bash
   cd /path/to/VisProbe
   pip install -e .
   ```

2. Check your Python environment:
   ```bash
   which python  # Verify you're using the right Python
   pip list | grep visprobe  # Check if installed
   ```

3. Reinstall in editable mode:
   ```bash
   pip uninstall visprobe
   pip install -e .
   ```

---

## Device and Memory Issues

### Problem: CUDA out of memory errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Force CPU mode:**
   ```python
   from visprobe import quick_check

   report = quick_check(model, data, preset="standard", device="cpu")
   ```

2. **Reduce test data size:**
   ```python
   # Test with fewer samples
   test_data = test_data[:50]  # Instead of all samples
   report = quick_check(model, test_data, preset="standard")
   ```

3. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Set environment variable:**
   ```bash
   export VISPROBE_DEVICE=cpu
   python your_script.py
   ```

### Problem: Device mismatch errors

**Symptoms:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
```

**Solutions:**

1. **Let VisProbe handle device management:**
   ```python
   # Use device="auto" (default)
   report = quick_check(model, data, preset="standard", device="auto")
   ```

2. **Ensure model and data are on same device:**
   ```python
   model = model.to("cuda")
   report = quick_check(model, data, preset="standard", device="cuda")
   ```

3. **Force CPU if unsure:**
   ```python
   model = model.to("cpu")
   report = quick_check(model, data, preset="standard", device="cpu")
   ```

### Problem: MPS (Apple Silicon) errors

**Symptoms:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**

1. **Force CPU:**
   ```python
   report = quick_check(model, data, preset="standard", device="cpu")
   ```

2. **Reduce batch size:**
   ```python
   test_data = test_data[:30]  # Smaller dataset
   ```

3. **Disable MPS:**
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

---

## Import and Module Errors

### Problem: `ImportError: cannot import name 'quick_check'`

**Symptoms:**
```
ImportError: cannot import name 'quick_check' from 'visprobe'
```

**Solutions:**

1. **Verify installation:**
   ```bash
   pip list | grep visprobe
   ```

2. **Reinstall:**
   ```bash
   pip uninstall visprobe
   pip install -e .
   ```

3. **Check Python path:**
   ```python
   import visprobe
   print(visprobe.__file__)  # Should point to your installation
   ```

## Data Format Issues

### Problem: `TypeError: unsupported data format`

**Symptoms:**
```
TypeError: Data must be DataLoader, TensorDataset, list of tuples, or tensor tuple
```

**Solutions:**

Convert your data to a supported format:

```python
# Option 1: List of (image, label) tuples
test_data = [(img1, label1), (img2, label2), ...]
report = quick_check(model, test_data, preset="standard")

# Option 2: Tuple of tensors
images = torch.stack([img1, img2, ...])
labels = torch.tensor([label1, label2, ...])
report = quick_check(model, (images, labels), preset="standard")

# Option 3: DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)
report = quick_check(model, loader, preset="standard")

# Option 4: TensorDataset
from torch.utils.data import TensorDataset
dataset = TensorDataset(images, labels)
report = quick_check(model, dataset, preset="standard")
```

### Problem: `RuntimeError: Expected 4D tensor (got 3D)`

**Symptoms:**
```
RuntimeError: Expected 4D input[NCHW] for conv2d, got 3D
```

**Solutions:**

Ensure your images are in `(C, H, W)` format, not `(H, W, C)`:

```python
# If images are (H, W, C), convert to (C, H, W)
if image.shape[-1] == 3:
    image = image.permute(2, 0, 1)

# Example with CIFAR-10
transform = T.Compose([
    T.ToTensor(),  # Converts (H, W, C) → (C, H, W) and scales to [0, 1]
])
```

### Problem: Image size mismatch

**Symptoms:**
```
RuntimeError: Input size mismatch
```

**Solutions:**

Resize images to match model's expected input size:

```python
import torchvision.transforms as T

# For ResNet (expects 224x224)
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
])

# Apply to dataset
dataset = ...
test_data = [(transform(img), label) for img, label in dataset]
```

---

## Normalization Issues

### Problem: Model predictions are wrong/random

**Symptoms:**
- Robustness score is unexpectedly low
- Model predictions don't match expected classes
- Results look random

**Solutions:**

1. **Check if model expects normalized inputs:**

   Most ImageNet pretrained models expect inputs normalized with:
   ```python
   mean = (0.485, 0.456, 0.406)
   std = (0.229, 0.224, 0.225)
   ```

   Pass these to `quick_check()`:
   ```python
   report = quick_check(
       model,
       data,
       preset="standard",
       mean=(0.485, 0.456, 0.406),
       std=(0.229, 0.224, 0.225)
   )
   ```

2. **If data is already normalized, don't pass mean/std:**
   ```python
   # Data already normalized in transform
   transform = T.Compose([
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
   ])

   # Don't pass mean/std to quick_check
   report = quick_check(model, data, preset="standard")
   ```

3. **Check normalization by testing a single image:**
   ```python
   # Test without VisProbe first
   test_img = test_data[0][0].unsqueeze(0).to(device)
   with torch.no_grad():
       output = model(test_img)
       pred = output.argmax(dim=1)
   print(f"Predicted class: {pred.item()}")
   ```

### Problem: `Warning: Data appears to be outside [0, 1] range`

**Symptoms:**
```
UserWarning: Input data appears to be outside [0, 1] range
```

**Solutions:**

This usually means your images are in [0, 255] range instead of [0, 1]:

```python
# If images are in [0, 255]
images = images / 255.0

# Or use ToTensor() transform
transform = T.ToTensor()  # Automatically scales [0, 255] → [0, 1]
```

---

## Performance Issues

### Problem: Testing is very slow

**Symptoms:**
- `quick_check()` takes a long time to complete

**Solutions:**

1. **Reduce budget:**
   ```python
   # Faster (less precise)
   report = quick_check(model, data, preset="standard", budget=100)

   # Normal (default)
   report = quick_check(model, data, preset="standard", budget=1000)

   # Slower (more precise)
   report = quick_check(model, data, preset="standard", budget=5000)
   ```

2. **Use fewer test samples:**
   ```python
   # Test with subset
   report = quick_check(model, data[:50], preset="standard")
   ```

3. **Use GPU:**
   ```python
   report = quick_check(model, data, preset="standard", device="cuda")
   ```

4. **Choose lighter preset:**
   ```python
   # Lighter: fewer strategies
   report = quick_check(model, data, preset="lighting")

   # Heavier: more strategies including compositional
   report = quick_check(model, data, preset="standard")
   ```

### Problem: High memory usage

**Solutions:**

1. **Process in smaller batches:**
   ```python
   # Split data into chunks
   chunk_size = 50
   all_failures = []

   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       report = quick_check(model, chunk, preset="standard")
       all_failures.extend(report.failures)
   ```

2. **Use CPU instead of GPU:**
   ```python
   report = quick_check(model, data, preset="standard", device="cpu")
   ```

---

## Result Interpretation

### Problem: Robustness score seems wrong

**Symptoms:**
- Score is unexpectedly high or low
- Doesn't match manual inspection

**Solutions:**

1. **Check normalization** (see [Normalization Issues](#normalization-issues))

2. **Inspect failures manually:**
   ```python
   report = quick_check(model, data, preset="standard")

   # Look at a few failures
   for failure in report.failures[:5]:
       print(f"Sample {failure['index']}")
       print(f"  Original: class {failure['original_pred']}")
       print(f"  Perturbed: class {failure['perturbed_pred']}")
       print(f"  Strategy: {failure['strategy']}")
       print(f"  Level: {failure['level']:.3f}")
   ```

3. **Export and visualize failures:**
   ```python
   path = report.export_failures(n=10)
   # Open images in path to see what's happening
   ```

4. **Test with known-good model:**
   ```python
   import torchvision.models as models
   model = models.resnet18(weights='IMAGENET1K_V1')
   model.eval()

   # Should get reasonable score (>0.6)
   report = quick_check(model, data, preset="standard")
   ```

### Problem: No failures found (score = 100%)

**Symptoms:**
- `report.score == 1.0`
- `len(report.failures) == 0`

**Possible causes:**

1. **Budget too low:**
   ```python
   # Increase budget to find more subtle failures
   report = quick_check(model, data, preset="standard", budget=2000)
   ```

2. **Model is genuinely robust** (rare)

3. **Wrong normalization** - model not actually working:
   ```python
   # Test a single prediction first
   with torch.no_grad():
       output = model(data[0][0].unsqueeze(0))
       print(f"Prediction: {output.argmax()}")
       print(f"Expected: {data[0][1]}")
   ```

### Problem: Too many failures (score < 30%)

**Symptoms:**
- Very low robustness score
- Almost all samples fail

**Possible causes:**

1. **Normalization mismatch** - Most likely!
   - Check if you need to pass `mean` and `std`
   - See [Normalization Issues](#normalization-issues)

2. **Model actually struggles with perturbations**
   - Try a lighter preset to isolate issues:
     ```python
     report = quick_check(model, data, preset="lighting", budget=500)
     ```

3. **Wrong device:**
   ```python
   # Ensure model is on correct device
   model = model.to("cuda")  # or "cpu"
   report = quick_check(model, data, preset="standard", device="cuda")
   ```

---

## Preset-Specific Issues

### Problem: "Preset not found" error

**Symptoms:**
```
ValueError: Unknown preset: my_preset
```

**Solutions:**

Check available presets:
```python
from visprobe import presets

for name, description in presets.list_presets():
    print(f"{name}: {description}")
```

Valid presets:
- `"standard"` (default)
- `"lighting"`
- `"blur"`
- `"corruption"`

---

## Debugging Tips

### Enable debug mode

```python
import os
os.environ["VISPROBE_DEBUG"] = "1"

from visprobe import quick_check
report = quick_check(model, data, preset="standard")
```

### Check VisProbe version

```python
import visprobe
print(visprobe.__version__)
```

### Minimal working example

Test with the simplest possible setup:

```python
import torch
import torchvision.models as models
from visprobe import quick_check

# Load pretrained model
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# Create dummy data
test_data = [(torch.randn(3, 224, 224), 0) for _ in range(10)]

# Test
report = quick_check(
    model,
    test_data,
    preset="standard",
    budget=100,
    device="cpu"
)

print(f"Score: {report.score:.1%}")
print(f"Failures: {len(report.failures)}")
```

If this works but your code doesn't, compare the differences.

---

## Getting More Help

1. **Check examples:**
   - See `examples/` directory for working code
   - Start with `examples/basic_example.py`

2. **Read API reference:**
   - See `COMPREHENSIVE_API_REFERENCE.md`

3. **Check main README:**
   - See `README.md` for overview

4. **Report bugs:**
   - Open an issue on GitHub with:
     - VisProbe version
     - Python version
     - PyTorch version
     - Full error traceback
     - Minimal reproduction code

---

## Common Error Messages

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `CUDA out of memory` | GPU memory full | Use `device="cpu"` or reduce data size |
| `Expected 4D tensor` | Wrong image format | Ensure images are `(C, H, W)` |
| `Device mismatch` | Model/data on different devices | Use `device="auto"` |
| `Score unexpectedly low` | Normalization mismatch | Pass correct `mean` and `std` |
| `ImportError: quick_check` | Old installation | Reinstall with `pip install -e .` |
| `Preset not found` | Typo in preset name | Use one of: standard, lighting, blur, corruption |

---

**Still stuck?** Open an issue with a minimal reproduction and we'll help!
