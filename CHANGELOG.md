# Changelog

All notable changes to Visfuzz will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-19

### Added
- Initial release of Visfuzz framework
- Core decorator-based API (`@given`, `@search`, `@model`, `@data_source`)
- Adversarial perturbation strategies:
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - BIM (Basic Iterative Method)
  - APGD (Auto-PGD)
  - Square Attack
- Natural perturbation strategies:
  - Gaussian Noise
  - Brightness adjustment
  - Rotation
- Robustness properties for classification:
  - LabelConstant (prediction stability)
  - TopKStability (top-k prediction consistency)
  - ConfidenceDrop (confidence degradation)
  - L2Distance (feature space distance)
- Interactive Streamlit dashboard for result visualization
- Command-line interface with `visprobe run` and `visprobe visualize` commands
- Comprehensive analysis utilities:
  - Ensemble analysis
  - Resolution impact analysis
  - Noise sensitivity analysis
  - Corruption robustness analysis
  - Layer-wise analysis
- Automatic device management (CPU/CUDA/MPS)
- Query counter for tracking model invocations
- Multi-property and composite strategy support
- Search modes: grid search, random search, adaptive search
- Complete API documentation and user guides
- 14 example test files demonstrating various use cases

### Documentation
- README.md with quick start guide
- COMPREHENSIVE_API_REFERENCE.md with complete API documentation
- API_OVERVIEW.md with architecture details
- DEVICE_MANAGEMENT.md for device configuration
- RELEASE_CHECKLIST.md for release validation
- TEST_DOCUMENTATION.md for test patterns

### Infrastructure
- MIT License
- Python package setup with setuptools
- Dependencies management via requirements.txt
- Entry point configuration for CLI tool

[0.1.0]: https://github.com/bilgedemirkaya/Visfuzz/releases/tag/v0.1.0
