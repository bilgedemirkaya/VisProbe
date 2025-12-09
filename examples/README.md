# VisProbe Examples

This directory contains examples showing how to use VisProbe to test your models.

## 🚀 Quick Start

If you're new to VisProbe, start with **`basic_example.py`**:

```bash
python examples/basic_example.py
```

This shows the absolute minimum code needed to test a model.

## 📚 Examples

### 1. **`basic_example.py`** - Start Here! ⭐
The simplest possible example. Tests a pretrained model with minimal code.

**What you'll learn:**
- How to use `quick_check()` in 3 lines
- How to view results with `report.show()`
- How to access results programmatically

**Run time:** ~30 seconds

```bash
python examples/basic_example.py
```

---

### 2. **`cifar10_example.py`** - Complete Workflow
A full end-to-end example using CIFAR-10 dataset.

**What you'll learn:**
- Loading real datasets
- Using correct normalization parameters
- Analyzing failure patterns
- Exporting failures for retraining

**Run time:** ~5-10 minutes (includes CIFAR-10 download)

```bash
python examples/cifar10_example.py
```

---

### 3. **`custom_model_example.py`** - Template for YOUR Model
A template you can copy and modify for your own model.

**What you'll learn:**
- How to adapt the code for your model
- Choosing the right preset for your use case
- Interpreting results for your specific application
- Best practices for robustness testing

**Run time:** Depends on your model

```bash
python examples/custom_model_example.py
```

---

### 4. **`preset_comparison.py`** - Compare All Presets
Tests your model with all 4 presets and compares results.

**What you'll learn:**
- Differences between presets
- Identifying your model's weak areas
- Understanding which perturbations are most challenging
- Making data-driven training decisions

**Run time:** ~10-15 minutes

```bash
python examples/preset_comparison.py
```

---

## 🎯 Which Example Should I Use?

| If you want to... | Use this example |
|-------------------|------------------|
| **Just see it work** | `basic_example.py` |
| **Test on CIFAR-10** | `cifar10_example.py` |
| **Test YOUR model** | `custom_model_example.py` |
| **Find weak areas** | `preset_comparison.py` |

## 📖 More Resources

- **Main README**: [`../README.md`](../README.md)
- **API Reference**: [`../COMPREHENSIVE_API_REFERENCE.md`](../COMPREHENSIVE_API_REFERENCE.md)
- **Troubleshooting**: [`../TROUBLESHOOTING.md`](../TROUBLESHOOTING.md)

## 💡 Tips

### Running Examples

```bash
# From project root:
python examples/basic_example.py

# Or make executable and run directly:
chmod +x examples/basic_example.py
./examples/basic_example.py
```

### Modifying for Your Needs

All examples are designed to be copy-paste friendly. Just:

1. Copy the example that's closest to your use case
2. Replace the model and data loading sections
3. Adjust the preset and normalization
4. Run and analyze!

### Common Modifications

**Change the preset:**
```python
report = quick_check(model, data, preset="blur")  # or "lighting", "corruption"
```

**Increase test budget (more thorough):**
```python
report = quick_check(model, data, preset="standard", budget=5000)
```

**Use GPU:**
```python
report = quick_check(model, data, preset="standard", device="cuda")
```

**Custom normalization:**
```python
report = quick_check(
    model, data, preset="standard",
    mean=(0.485, 0.456, 0.406),  # Your mean
    std=(0.229, 0.224, 0.225)     # Your std
)
```

## 🐛 Troubleshooting

### "CUDA out of memory"
- Use `device="cpu"` or reduce the number of test samples

### "RuntimeError: Expected 4D tensor"
- Make sure your images are shaped `(C, H, W)` or `(N, C, H, W)`
- Check that your data format matches what VisProbe expects

### "Normalization warning"
- Set `mean` and `std` to match your training normalization
- See `custom_model_example.py` for examples

## 🤝 Contributing Examples

Have a cool use case? We'd love to add it!

Good examples to contribute:
- Object detection models
- Specific domains (medical, satellite, etc.)
- Advanced configurations
- Integration with training pipelines

Open a PR or issue!
