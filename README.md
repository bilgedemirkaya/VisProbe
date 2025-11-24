# VisProbe: Adversarial Robustness Testing and Visualization

VisProbe is a Python framework for testing the robustness of neural networks against adversarial attacks. It provides a simple decorator-based API for defining tests and a powerful, interactive dashboard for visualizing the results.

## Key Features

*   **Simple, Declarative API**: Define complex robustness tests with a few simple decorators (`@given`, `@search`, `@model`, `@data_source`).
*   **Interactive Dashboard**: Visualize test results, compare original and perturbed images, and dive deep into model behavior with detailed metrics.
*   **Built-in Analyses**: Automatically get insights into resolution impact, noise sensitivity, intermediate layer analysis, and top-k prediction stability.
*   **Extensible**: Easily add your own perturbation strategies and robustness properties.

## Getting Started

1.  **Installation**:
    ```bash
    pip install -e .
    ```

2.  **Write a Test**: Create a Python file (e.g., `test_my_model.py`) and use the VisProbe decorators to define your test.

    ```python
    # test_my_model.py
    from visprobe import given, model, data_source
    from visprobe.strategies import FGSMStrategy
    from visprobe.properties import LabelConstant

    # ... (load your model and data) ...

    @given(strategy=FGSMStrategy(eps=0.01))
    @model(my_model)
    @data_source(data_obj=my_data, collate_fn=my_collate_fn)
    def test_robustness(original, perturbed):
        assert LabelConstant.evaluate(original, perturbed)
    ```

3.  **Visualize the Results**:
    ```bash
    visprobe visualize test_my_model.py
    ```
    This command will automatically run your test and launch the interactive dashboard in your browser.

## The VisProbe Dashboard

The dashboard provides a comprehensive overview of your model's robustness. Here's a breakdown of the key metrics and visualizations:

### Key Metrics

*   **Robust Accuracy** (`@given` tests): The percentage of samples that passed the robustness property.
*   **Failure Threshold** (`@search` tests): The minimum perturbation level (`ε`) required to cause a test failure.
*   **Model Queries**: The total number of times the model was queried during the test.
*   **Runtime**: The total time taken to run the test.

## Available Strategies

### Adversarial Attacks
- **FGSMStrategy**: Fast Gradient Sign Method
- **PGDStrategy**: Projected Gradient Descent (stronger iterative attack)
- **BIMStrategy**: Basic Iterative Method
- **APGDStrategy**: Auto-Projected Gradient Descent
- **SquareAttackStrategy**: Score-based black-box attack

### Natural Perturbations
- **GaussianNoiseStrategy**: Additive Gaussian noise
- **BrightnessStrategy**: Brightness adjustment
- **RotateStrategy**: Image rotation

## Available Properties

- **LabelConstant**: Top-1 prediction must remain unchanged
- **TopKStability**: Top-k predictions overlap analysis
- **ConfidenceDrop**: Confidence decrease must stay within limits
- **L2Distance**: L2 distance between output vectors must be bounded

## Installation

### Basic Installation
```bash
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

### With Enhanced Visualizations
```bash
pip install -e ".[viz]"
```

## Security Considerations

⚠️ **IMPORTANT: Please read before using VisProbe**

### Model Loading Security

VisProbe loads and executes PyTorch models, which can contain arbitrary Python code. **Only load models from trusted sources.**

**Risks:**
- PyTorch's `torch.load()` uses pickle, which can execute arbitrary code
- Malicious models can compromise your system, steal data, or execute harmful operations
- Models from untrusted sources should be treated as potentially dangerous

**Best Practices:**
1. **Only use models from trusted sources** (official model zoos, verified researchers, your own trained models)
2. **Never load models from unknown or untrusted sources**
3. **Use `torch.load()` with `weights_only=True` when possible** for models that support it
4. **Inspect model code before loading** if available as source
5. **Run tests in isolated environments** (containers, VMs) when testing untrusted models
6. **Keep PyTorch and dependencies updated** to get the latest security patches

### Data Security

- Test results may contain sensitive information about model vulnerabilities
- Store test results securely and limit access appropriately
- Be cautious when sharing test results publicly, as they may reveal attack vectors

### Additional Security Notes

- VisProbe executes user-provided test files as Python scripts
- Results are saved to `/tmp/visprobe_results` by default (configurable via `VISPROBE_RESULTS_DIR`)
- The CLI validates file paths to prevent path traversal attacks
- See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities

## Quick Start Examples

### Basic Adversarial Testing
```python
import visprobe.auto_init  # Automatic device management
from visprobe import given, model, data_source
from visprobe.strategies import FGSMStrategy
from visprobe.properties import LabelConstant

@given(strategy=FGSMStrategy(eps=0.031))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_fgsm_robustness(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)

# Run the test
if __name__ == "__main__":
    test_fgsm_robustness()
```

### Adaptive Threshold Search
```python
from visprobe import search
from visprobe.strategies import FGSMStrategy

@search(
    strategy=lambda level: FGSMStrategy(eps=level),
    initial_level=0.001,
    step=0.005,
    max_queries=100
)
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def find_failure_threshold(original, perturbed):
    return LabelConstant.evaluate(original, perturbed)
```

### Multiple Properties
```python
from visprobe.properties import LabelConstant, ConfidenceDrop, TopKStability

@given(strategy=FGSMStrategy(eps=0.031))
@model(my_model)
@data_source(data_obj=my_data, collate_fn=my_collate_fn)
def test_comprehensive_robustness(original, perturbed):
    # All conditions must pass
    assert LabelConstant.evaluate(original, perturbed)
    assert ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3)
    assert TopKStability.evaluate(original, perturbed, k=5, min_overlap=3)
```

## CLI Usage

```bash
# Run tests and save results
visprobe run my_test.py

# Launch interactive dashboard
visprobe visualize my_test.py

# Specify device
visprobe run my_test.py --device cuda
visprobe visualize my_test.py --device cpu
```

## Device Management

VisProbe provides automatic device management to prevent common issues:

```python
import visprobe.auto_init  # Add this line to your test files
```

Or configure manually:
```bash
export VISPROBE_DEVICE=cpu        # Force CPU
export VISPROBE_DEVICE=cuda       # Force CUDA
export VISPROBE_PREFER_GPU=1      # Prefer GPU if available
```

## Documentation

- **[Comprehensive API Reference](COMPREHENSIVE_API_REFERENCE.md)**: Complete API documentation
- **[Device Management Guide](DEVICE_MANAGEMENT.md)**: Device configuration and troubleshooting
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Common issues and solutions
- **[Performance Guide](PERFORMANCE.md)**: Optimization tips and benchmarks
- **[API Architecture Overview](API_OVERVIEW.md)**: Internal architecture details
- **[Test Documentation](examples/TEST_DOCUMENTATION.md)**: Example test patterns and use cases
- **[Changelog](CHANGELOG.md)**: Version history and changes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use VisProbe in your research, please cite:

```bibtex
@software{visprobe2025,
  title={VisProbe: Interactive Robustness Testing for Computer Vision Models},
  author={VisProbe Development Team},
  year={2025},
  url={https://github.com/bilgedemirkaya/VisProbe}
}
```

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/bilgedemirkaya/VisProbe/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/bilgedemirkaya/VisProbe/discussions)
- **Documentation**: Full documentation available in the repository

## Dashboard Features

### Visual Comparison
A side-by-side comparison of the original and perturbed images, along with the model's prediction and confidence for each.

### Detailed Analysis Tabs
*   **Search Path** (`@search` tests): Interactive chart showing how model confidence and prediction change as perturbation level increases
*   **Ensemble Analysis**: Bar chart showing cosine similarity of intermediate layer activations between original and perturbed images
*   **Resolution Impact**: Chart showing how model robustness changes at different input resolutions
*   **Noise Sensitivity**: Chart showing how model accuracy degrades as Gaussian noise is added to input
*   **Top-K Overlap**: Chart showing overlapping predictions in top-K set between original and perturbed images
*   **Raw Report**: Full JSON report for the test, available for download

## Troubleshooting

### Common Issues
- **Device mismatch errors**: Use `import visprobe.auto_init` or set `VISPROBE_DEVICE=cpu`
- **Memory issues**: Reduce batch size or use smaller datasets for testing
- **Missing ART**: Install with `pip install adversarial-robustness-toolbox`
- **Streamlit not found**: Install with `pip install streamlit>=1.28.0`

### Getting Help
- Check the [Device Management Guide](DEVICE_MANAGEMENT.md) for device-related issues
- Review example test files in the repository
- Enable debug mode with `export VISPROBE_DEBUG=1`
