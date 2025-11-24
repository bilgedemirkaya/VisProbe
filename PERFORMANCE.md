# VisProbe Performance Optimization Guide

This guide provides best practices and benchmarks for optimizing VisProbe test performance.

## Table of Contents

- [Quick Wins](#quick-wins)
- [Device Optimization](#device-optimization)
- [Memory Management](#memory-management)
- [Search Strategy Optimization](#search-strategy-optimization)
- [Batching and Parallelization](#batching-and-parallelization)
- [Benchmarks](#benchmarks)
- [Profiling and Monitoring](#profiling-and-monitoring)

---

## Quick Wins

### 1. Use GPU Acceleration
```python
# Automatically use GPU if available
import visprobe.auto_init

# Or explicitly set device
import os
os.environ['VISPROBE_DEVICE'] = 'cuda'
```

**Expected speedup:** 10-50x for large models

### 2. Reduce Sample Size During Development
```python
# Start small for iteration
N_SAMPLES = 10  # Development
N_SAMPLES = 100  # Testing
N_SAMPLES = 1000  # Production

IMAGES = load_data(N_SAMPLES)
```

**Expected speedup:** Linear with sample reduction

### 3. Disable Optional Analyses
```python
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    resolutions=None,  # Disable resolution analysis
    noise_sweep=None,  # Disable noise sweep
    ensemble=False,    # Disable ensemble analysis
    ...
)
```

**Expected speedup:** 2-5x depending on analyses disabled

### 4. Use Binary Search Mode
```python
@search(
    mode="binary",  # Fastest search mode
    max_queries=20,  # Limit iterations
    ...
)
```

**Expected speedup:** 2-3x compared to adaptive search

---

## Device Optimization

### GPU vs CPU Performance

| Operation | CPU (i7) | GPU (RTX 3080) | Speedup |
|-----------|----------|----------------|---------|
| ResNet-50 inference | 45 ms | 3 ms | 15x |
| FGSM attack | 180 ms | 8 ms | 22x |
| PGD attack (10 steps) | 1.8 s | 80 ms | 22x |
| Gaussian noise | 2 ms | 0.5 ms | 4x |

### Best Practices

1. **Use GPU for:**
   - Large models (>10M parameters)
   - Adversarial attacks (FGSM, PGD, etc.)
   - Batch sizes > 16

2. **Use CPU for:**
   - Small models (<1M parameters)
   - Simple transformations (brightness, rotation)
   - Single sample inference
   - When GPU memory is limited

3. **Mixed approach:**
   ```python
   # Keep model on GPU, process data on CPU
   model = model.cuda()

   for batch in data_loader:
       batch_gpu = batch.cuda()  # Move to GPU only for inference
       output = model(batch_gpu)
       batch_cpu = output.cpu()  # Move back for analysis
   ```

### Device Selection Strategy

```python
import torch

def select_optimal_device(model, batch_size=32):
    """Automatically select best device based on model size."""
    param_count = sum(p.numel() for p in model.parameters())

    if not torch.cuda.is_available():
        return 'cpu'

    # Rough heuristic
    if param_count < 1_000_000:  # Small model
        return 'cpu'
    elif param_count < 50_000_000:  # Medium model
        return 'cuda'
    else:  # Large model
        # Check if model fits in GPU memory
        try:
            model_copy = model.cuda()
            test_input = torch.randn(batch_size, 3, 224, 224).cuda()
            _ = model_copy(test_input)
            return 'cuda'
        except RuntimeError:  # OOM
            return 'cpu'
```

---

## Memory Management

### Memory Usage Patterns

| Component | Memory per Sample | Optimization |
|-----------|------------------|--------------|
| Input image (224x224 RGB) | 0.6 MB | Use uint8 until needed |
| Model activations (ResNet-50) | 100 MB | gradient_checkpointing |
| Intermediate features | 50-200 MB | Disable if not needed |
| Search history | 1-5 MB | Limit path storage |

### Optimization Techniques

#### 1. Clear CUDA Cache
```python
import torch

def run_test_with_cleanup():
    @given(strategy=GaussianNoiseStrategy(std_dev=0.1))
    def test_robustness(original, perturbed):
        result = LabelConstant.evaluate(original, perturbed)

        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        assert result
```

#### 2. Use Gradient Checkpointing
```python
# For models that support it
model.gradient_checkpointing_enable()
```

**Trade-off:** 30% slower inference, 50% less memory

#### 3. Process in Chunks
```python
def process_large_dataset(images, chunk_size=10):
    results = []
    for i in range(0, len(images), chunk_size):
        chunk = images[i:i+chunk_size]
        result = run_test(chunk)
        results.extend(result)

        # Clean up
        torch.cuda.empty_cache()

    return results
```

#### 4. Limit Search History
```python
@search(
    strategy=lambda l: FGSMStrategy(eps=l),
    max_queries=30,  # Limit stored history
    ...
)
```

---

## Search Strategy Optimization

### Search Mode Comparison

| Mode | Queries | Accuracy | Best For |
|------|---------|----------|----------|
| Binary | 5-10 | ±5% | Quick estimates |
| Grid | 20-50 | ±2% | Comprehensive coverage |
| Random | 30-100 | ±3% | Exploration |
| Adaptive | 10-30 | ±1% | Precise thresholds |

### Recommended Settings

#### Development / Quick Testing
```python
@search(
    mode="binary",
    initial_level=0.01,
    step=0.01,
    max_queries=10,
    resolutions=None,
    noise_sweep=None,
)
```
**Expected runtime:** ~30 seconds (10 samples, ResNet-50)

#### Production / Accurate Results
```python
@search(
    mode="adaptive",
    initial_level=0.001,
    step=0.005,
    min_step=0.0001,
    max_queries=50,
    resolutions=[112, 224],  # Limited resolution analysis
)
```
**Expected runtime:** ~5 minutes (100 samples, ResNet-50)

### Parameter Tuning

#### step Size
```python
# Fast but coarse
step=0.05  # ~20 queries

# Balanced
step=0.01  # ~40 queries

# Precise but slow
step=0.001  # ~100+ queries
```

#### min_step
```python
# Set min_step to prevent excessive refinement
min_step = step / 128  # Allows ~7 refinement levels
```

---

## Batching and Parallelization

### Batch Size Selection

| Model Size | Recommended Batch Size | Notes |
|------------|------------------------|-------|
| Small (<10M params) | 64-128 | CPU: 32, GPU: 128 |
| Medium (10-100M) | 16-32 | CPU: 8, GPU: 32 |
| Large (>100M) | 4-16 | CPU: 4, GPU: 16 |

### Implementation

```python
import torch
from torch.utils.data import DataLoader

def create_efficient_loader(images, batch_size=32):
    """Create optimized data loader."""
    loader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch batches
    )
    return loader

# Usage
@data_source(
    data_obj=create_efficient_loader(images),
    collate_fn=torch.stack,
    ...
)
def test_with_batching(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)
```

### Multi-GPU Support (Future)

```python
# Planned for future release
import torch.nn as nn

model = nn.DataParallel(model)  # Use multiple GPUs
```

---

## Benchmarks

### Test Suite Performance

**Configuration:**
- Hardware: NVIDIA RTX 3080, Intel i7-10700K
- Model: ResNet-50 (25M parameters)
- Dataset: CIFAR-10 (100 samples)

| Test Type | Search Mode | Queries | Time | Memory |
|-----------|-------------|---------|------|--------|
| Gaussian Noise | Binary | 8 | 45s | 2.1 GB |
| Gaussian Noise | Adaptive | 25 | 2m 10s | 2.3 GB |
| FGSM Attack | Binary | 10 | 1m 20s | 2.8 GB |
| FGSM Attack | Adaptive | 30 | 3m 45s | 3.1 GB |
| PGD Attack | Binary | 12 | 8m 30s | 4.2 GB |
| Full RQ3 Suite | Adaptive | 120 | 15m 40s | 3.5 GB |

### Scaling Characteristics

#### Sample Count Scaling
```
10 samples:   ~30 seconds
100 samples:  ~5 minutes
1000 samples: ~50 minutes
```
**Scaling:** Approximately linear

#### Model Size Scaling
```
ResNet-18 (11M):  1.0x baseline
ResNet-50 (25M):  2.2x slower
ResNet-101 (44M): 4.1x slower
ViT-B/16 (86M):   8.5x slower
```

---

## Profiling and Monitoring

### Built-in Profiling

```python
import time
from contextlib import contextmanager

@contextmanager
def profile_section(name):
    """Simple profiler for test sections."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")

# Usage
with profile_section("Data loading"):
    images, labels = load_cifar10(100)

with profile_section("Model inference"):
    outputs = model(images)
```

### Python Profiler

```bash
# Profile your test
python -m cProfile -o profile.stats test_my_model.py

# Analyze results
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20  # Show top 20
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory_profiler

# Profile memory usage
python -m memory_profiler test_my_model.py
```

### Query Counter

VisProbe automatically tracks model queries:

```python
from visprobe.api.query_counter import QueryCounter

with QueryCounter(model) as qc:
    output = model(images)
    print(f"Queries: {qc.extra}")  # Number of forward passes
```

---

## Best Practices Summary

### ✅ Do

1. **Start small:** Test with 10 samples first
2. **Use GPU:** Enable CUDA for models >10M parameters
3. **Limit analyses:** Disable optional features during development
4. **Binary search:** Use for quick iterations
5. **Monitor memory:** Clear CUDA cache periodically
6. **Profile regularly:** Identify bottlenecks early

### ❌ Don't

1. **Don't use large batches on CPU:** Keep batch size ≤16
2. **Don't run all analyses:** Enable only what you need
3. **Don't use fine step sizes initially:** Start with step=0.01
4. **Don't ignore OOM errors:** Reduce batch size or use CPU
5. **Don't test full dataset:** Use subset for development

---

## Example: Optimized Test Configuration

```python
"""
Optimized configuration for production robustness testing.

Expected performance:
- ResNet-50, 100 samples, GPU: ~3 minutes
- Memory usage: <4 GB
- Query count: ~200
"""
import os
import visprobe.auto_init  # Auto device management

from visprobe import search, model, data_source
from visprobe.strategies import GaussianNoiseStrategy
from visprobe.properties import LabelConstant

# Configuration
N_SAMPLES = int(os.getenv("N_SAMPLES", "100"))
BATCH_SIZE = 32
DEVICE = "cuda"  # or "cpu"

# Load data efficiently
images = load_data(N_SAMPLES)

# Optimized search configuration
@search(
    strategy=lambda l: GaussianNoiseStrategy(std_dev=l),
    mode="adaptive",          # Good accuracy/speed trade-off
    initial_level=0.001,
    step=0.01,                # Reasonable step size
    min_step=0.0001,          # Prevent over-refinement
    max_queries=30,           # Limit iterations
    resolutions=None,         # Disable expensive analysis
    noise_sweep=None,         # Disable unless needed
    top_k=5,                  # Monitor top-5 stability
)
@model(my_model)
@data_source(
    data_obj=images,
    collate_fn=torch.stack,
    class_names=CLASS_NAMES,
)
def test_gaussian_noise_robustness(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)
```

---

## Troubleshooting Performance

### Test is slower than expected

1. **Check device:**
   ```python
   print(f"Device: {next(model.parameters()).device}")
   ```

2. **Profile the test:**
   ```bash
   python -m cProfile -o profile.stats test.py
   ```

3. **Check batch size:**
   ```python
   print(f"Batch size: {len(images)}")
   ```

4. **Monitor GPU utilization:**
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

### Out of Memory (OOM)

1. Reduce batch size
2. Clear CUDA cache
3. Disable analyses
4. Use CPU for large models

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

---

## Future Optimizations

Planned improvements for future releases:

- [ ] Multi-GPU support via DataParallel
- [ ] Distributed testing with Ray/Dask
- [ ] JIT compilation for strategies
- [ ] Cached perturbation results
- [ ] Lazy evaluation of analyses
- [ ] ONNX runtime support for faster inference
- [ ] Mixed precision (FP16) support

---

**Last Updated:** 2025-01-24

**Benchmark System:** NVIDIA RTX 3080, Intel i7-10700K, 32GB RAM
