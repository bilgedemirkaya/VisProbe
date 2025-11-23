# VisProbe Device Management

VisProbe provides automatic device management to prevent common device mismatch errors and ensure stable execution across different platforms.

## Quick Start (Recommended)

Add this single import at the top of your test files:

```python
import visprobe.auto_init  # Automatic device/threading configuration

# Your test code - no manual device management needed
from visprobe.api.decorators import given, model, data_source
from visprobe.properties.classification import LabelConstant
# ...
```

This automatically:
- Configures stable device selection (defaults to CPU for maximum compatibility)
- Sets up optimal threading for performance and stability
- Suppresses common noisy warnings
- Handles model and data device placement

## Manual Configuration (Advanced)

If you need more control, you can use environment variables:

```bash
# Force specific device
export VISPROBE_DEVICE=cpu        # Force CPU
export VISPROBE_DEVICE=cuda       # Force CUDA
export VISPROBE_DEVICE=cuda:1     # Specific CUDA device

# Enable GPU preference (will use CUDA if available)
export VISPROBE_PREFER_GPU=1

# Control threading
export VF_THREADS=4              # PyTorch threads
export OMP_NUM_THREADS=4         # OpenMP threads
```

## Device Selection Strategy

VisProbe uses this priority order for device selection:

1. **Explicit `VISPROBE_DEVICE`** environment variable
2. **CUDA** if available and `VISPROBE_PREFER_GPU=1` is set
3. **CPU** (default for maximum stability)

**Note**: MPS (Apple Metal) is intentionally deprioritized due to frequent device mismatch issues with mixed PyTorch/external library usage (like ART for adversarial attacks).

## Migration Guide

### Before (manual device management in every file):
```python
import os
import torch
os.environ["VISPROBE_DEVICE"] = "cpu"
torch.set_num_threads(1)

my_model = load_model()
my_model = my_model.cpu()  # Manual device placement
```

### After (automatic):
```python
import visprobe.auto_init  # One line handles everything

my_model = load_model()  # Device placement handled automatically
```

## Troubleshooting

### Device Mismatch Errors
If you see errors like `Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same`:

1. Add `import visprobe.auto_init` at the top of your file
2. Or set `export VISPROBE_DEVICE=cpu`

### Performance Optimization
For GPU usage when stable:
```bash
export VISPROBE_PREFER_GPU=1
export VF_THREADS=4
```

### Debug Device Selection
```bash
export VISPROBE_DEBUG=1  # Shows device selection info
python your_test.py
```
