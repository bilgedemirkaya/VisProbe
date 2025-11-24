# VisProbe Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using VisProbe.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Device and Memory Issues](#device-and-memory-issues)
- [Import and Module Errors](#import-and-module-errors)
- [Test Execution Issues](#test-execution-issues)
- [Dashboard and Visualization Issues](#dashboard-and-visualization-issues)
- [Performance Issues](#performance-issues)
- [Security and Permissions](#security-and-permissions)

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
   import visprobe.auto_init  # Add at top of test file
   # Or set environment variable
   export VISPROBE_DEVICE=cpu
   ```

2. **Reduce batch size:**
   ```python
   # Limit number of samples
   N_SAMPLES = 10  # Instead of 256
   ```

3. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Use gradient checkpointing** (for large models):
   ```python
   model.gradient_checkpointing_enable()  # If supported
   ```

### Problem: Device mismatch errors

**Symptoms:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Solutions:**

1. **Use auto_init** (recommended):
   ```python
   import visprobe.auto_init  # Automatically handles device placement
   ```

2. **Explicitly set device:**
   ```bash
   export VISPROBE_DEVICE=cpu
   # or
   export VISPROBE_DEVICE=cuda
   ```

3. **Check model and data devices:**
   ```python
   print(f"Model device: {next(model.parameters()).device}")
   print(f"Data device: {images[0].device}")
   ```

### Problem: MPS (Apple Silicon) errors

**Symptoms:**
```
RuntimeError: MPS backend out of memory
```

**Solutions:**
```python
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS caching
```

Or force CPU:
```bash
export VISPROBE_DEVICE=cpu
```

---

## Import and Module Errors

### Problem: `ImportError: cannot import name 'X' from 'visprobe'`

**Solutions:**

1. Check the correct import path:
   ```python
   # Correct imports
   from visprobe import given, search, model, data_source
   from visprobe.strategies import GaussianNoiseStrategy, FGSMStrategy
   from visprobe.properties import LabelConstant, TopKStability
   ```

2. Verify installation:
   ```bash
   python -c "import visprobe; print(visprobe.__file__)"
   ```

3. Reinstall package:
   ```bash
   pip install -e . --force-reinstall --no-cache-dir
   ```

### Problem: `AttributeError: module 'visprobe' has no attribute 'X'`

**Solution:** Check the API documentation. The attribute might have been moved or renamed.

---

## Test Execution Issues

### Problem: Tests hang or run indefinitely

**Possible Causes:**
1. Search space too large
2. Property never fails
3. Infinite loop in strategy

**Solutions:**

1. **Set max_queries limit:**
   ```python
   @search(
       strategy=lambda l: FGSMStrategy(eps=l),
       max_queries=50,  # Limit iterations
       ...
   )
   ```

2. **Add timeout:**
   ```bash
   timeout 300 python test_my_model.py  # 5 minute timeout
   ```

3. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Problem: All tests fail immediately

**Solutions:**

1. **Check property definition:**
   ```python
   # Make sure property is callable
   def test_robustness(original, perturbed):
       assert LabelConstant.evaluate(original, perturbed)  # Correct
       # NOT: assert LabelConstant(original, perturbed)
   ```

2. **Verify data format:**
   ```python
   # Images should be tensors
   print(type(images[0]))  # Should be torch.Tensor
   print(images[0].shape)  # Should be [C, H, W] or [B, C, H, W]
   ```

3. **Check model output format:**
   ```python
   with torch.no_grad():
       output = model(images)
       print(output.shape)  # Should be [B, num_classes]
   ```

### Problem: `ValidationError` when using decorators

**Symptoms:**
```
ValidationError: Test function 'my_test' must accept at least 2 parameters
```

**Solution:** Ensure your test function has the correct signature:
```python
# Correct
@given(strategy=GaussianNoiseStrategy(std_dev=0.1))
def test_robustness(original, perturbed):  # ✓ 2 parameters
    assert LabelConstant.evaluate(original, perturbed)

# Incorrect
def test_robustness(data):  # ✗ Only 1 parameter
    pass
```

---

## Dashboard and Visualization Issues

### Problem: `streamlit: command not found`

**Solution:**
```bash
pip install streamlit>=1.28.0
```

### Problem: Dashboard shows "No results found"

**Solutions:**

1. **Check results directory:**
   ```bash
   ls /tmp/visprobe_results/  # On Unix/Mac
   echo %TEMP%\visprobe_results  # On Windows
   ```

2. **Run test first:**
   ```bash
   visprobe run test_my_model.py
   visprobe visualize test_my_model.py
   ```

3. **Set custom results directory:**
   ```bash
   export VISPROBE_RESULTS_DIR=/path/to/results
   python test_my_model.py
   ```

### Problem: Images not displaying in dashboard

**Solutions:**

1. Check if images were saved:
   ```bash
   ls /tmp/visprobe_results/*.png
   ```

2. Verify mean/std parameters:
   ```python
   @data_source(
       data_obj=images,
       mean=[0.485, 0.456, 0.406],  # Must match preprocessing
       std=[0.229, 0.224, 0.225]
   )
   ```

---

## Performance Issues

### Problem: Tests are very slow

**Solutions:**

1. **Reduce sample size:**
   ```python
   N_SAMPLES = 10  # Start small for debugging
   ```

2. **Disable expensive analyses:**
   ```python
   @search(
       strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
       resolutions=None,  # Disable resolution analysis
       noise_sweep=None,  # Disable noise sweep
       ...
   )
   ```

3. **Use faster search mode:**
   ```python
   @search(
       mode="binary",  # Faster than adaptive
       ...
   )
   ```

4. **Profile your code:**
   ```bash
   python -m cProfile -o profile.stats test_my_model.py
   python -m pstats profile.stats
   ```

See [PERFORMANCE.md](PERFORMANCE.md) for detailed optimization guide.

---

## Security and Permissions

### Problem: Permission denied when saving results

**Solutions:**

1. **Set writable directory:**
   ```bash
   export VISPROBE_RESULTS_DIR=$HOME/visprobe_results
   mkdir -p $HOME/visprobe_results
   ```

2. **Check permissions:**
   ```bash
   ls -ld /tmp/visprobe_results/
   ```

### Problem: Model loading security warning

**This is expected.** PyTorch's `torch.load()` can execute arbitrary code. Only load models from trusted sources.

**Recommended practices:**
1. Only use models from official sources
2. Use `torch.load(path, weights_only=True)` when possible
3. Inspect model source code before loading
4. Run tests in isolated environments (containers/VMs) for untrusted models

---

## Common Error Messages

### `TypeError: unhashable type: 'list'`

**Cause:** Passing list to a function expecting hashable type.

**Solution:** Check property evaluation - might need to convert lists to tuples.

### `AssertionError` with no message

**Cause:** Property test failed without explanation.

**Solution:** Add descriptive assertion messages:
```python
def test_robustness(original, perturbed):
    result = LabelConstant.evaluate(original, perturbed)
    assert result, f"LabelConstant failed: prediction changed from {original} to {perturbed}"
```

### `RuntimeError: Expected 4D tensor but got 3D`

**Cause:** Missing batch dimension.

**Solution:**
```python
# Add batch dimension if needed
if images.dim() == 3:
    images = images.unsqueeze(0)
```

---

## Getting Help

If you're still experiencing issues:

1. **Check existing issues:** [GitHub Issues](https://github.com/bilgedemirkaya/VisProbe/issues)
2. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. **Create a minimal reproducible example**
4. **Report the issue** with:
   - VisProbe version: `pip show visprobe`
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Operating system
   - Full error traceback
   - Minimal code to reproduce

---

## Debugging Tips

### Enable verbose output
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check environment variables
```python
import os
print(f"VISPROBE_DEVICE: {os.getenv('VISPROBE_DEVICE')}")
print(f"VISPROBE_RESULTS_DIR: {os.getenv('VISPROBE_RESULTS_DIR')}")
print(f"VISPROBE_DEBUG: {os.getenv('VISPROBE_DEBUG')}")
```

### Verify tensor shapes and devices
```python
def debug_tensor(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

debug_tensor("Input", images)
debug_tensor("Output", model_output)
```

---

**Last Updated:** 2025-01-24
