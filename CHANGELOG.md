# Changelog

All notable changes to VisProbe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Threat-Model-Aware Preset System ⭐

**Major Feature: Threat-Model-Aware Presets**
- New preset architecture distinguishing between three threat models:
  - **`natural`** - Environmental perturbations (passive threat model)
  - **`adversarial`** - Gradient-based attacks (active threat model)
  - **`realistic_attack`** - Attacks under suboptimal conditions (active + environmental) - **KEY CONTRIBUTION**
  - **`comprehensive`** - Complete evaluation across all threat models
- Opportunistic vulnerability detection: `_check_opportunistic_vulnerability()`
- Per-threat-model robustness scoring with vulnerability warnings
- New `compare_threat_models()` convenience function for comparing all presets
- Adversarial strategy support: FGSM, PGD, BIM
- Strategy categorization for comprehensive preset breakdown

**Report Enhancements:**
- `threat_model` property - Returns the threat model used for testing
- `threat_model_scores` property - Per-threat-model breakdown
- `threat_model_summary` property - Comprehensive analysis with vulnerability detection
- `vulnerability_warning` property - Critical security warnings for opportunistic attacks
- Updated display methods for threat model visualization:
  - `_show_text()` - Console display with threat breakdown
  - `_generate_html_summary()` - Jupyter display with colored cards
  - `_generate_html_full()` - Browser report with breakdown section

**Backward Compatibility:**
- Legacy presets (standard, lighting, blur, corruption) maintained with deprecation warnings
- Automatic migration hints for users

**Other Additions:**
- Shared CLI utilities module (`cli/utils.py`)
- Helper functions for search path building and strategy configuration
- Comprehensive docstrings for internal methods
- Debug logging for exception handling
- **New modular architecture for runner.py**:
  - `api/context.py` - Test context initialization and device management
  - `api/property_evaluator.py` - Property evaluation logic
  - `api/strategy_utils.py` - Strategy configuration and serialization
  - `api/report_builder.py` - Report construction utilities
- Comprehensive troubleshooting guide (TROUBLESHOOTING.md)
- Performance optimization guide (PERFORMANCE.md)

### Changed
- Consolidated package configuration to `pyproject.toml` only (removed `setup.py`)
- Moved examples from `src/test-files/` to `examples/` directory
- Improved normalization functions to use shared `utils.to_image_space()` and `utils.to_model_space()`
- Enhanced exception handling with informative debug messages
- Updated all documentation links to reflect new structure
- **[MAJOR] Refactored runner.py into focused modules** (reduced from 744 to 326 lines):
  - Extracted context initialization into `TestContext` class
  - Extracted property evaluation into `PropertyEvaluator` class
  - Extracted strategy utilities into `StrategyConfig` class
  - Extracted report building into `ReportBuilder` class
  - Improved separation of concerns and maintainability

### Fixed
- **[SECURITY]** Fixed path traversal vulnerability in report generation (sanitized filenames)
- **[SECURITY]** Fixed command injection vulnerability in CLI (validated file paths)
- Removed duplicate `get_results_dir()` implementations
- Removed unused imports and variables (resize, clean_feat, pert_feat, fail_idx)
- Improved search path building logic (eliminated ~60 lines of duplication)
- Consolidated strategy configuration pattern (eliminated ~45 lines of duplication)

### Removed
- `setup.py` (consolidated to pyproject.toml)
- Unused torchvision.resize import from runner.py
- Duplicate get_results_dir() from cli.py and dashboard_helpers.py
- ~115 lines of duplicated code through helper function consolidation

### Security
- Added `_sanitize_filename()` to prevent path traversal in reports
- Added `_validate_test_file()` to prevent command injection in CLI
- Added comprehensive security warnings to README.md
- Updated SECURITY.md with real contact information (bilgedemirkaya@example.com)

### Documentation
- Added Security Considerations section to README
- Updated MANIFEST.in to reference new structure
- Updated .gitignore for examples/ directory
- Enhanced inline documentation with detailed docstrings

### Performance
- Reduced codebase by ~129 lines through deduplication
- Optimized imports (removed unused dependencies)
- Created reusable helper functions for common patterns
- **[ARCHITECTURE] Improved maintainability with modular design**:
  - runner.py reduced by 56% (744 → 326 lines)
  - Better code organization across 5 focused modules
  - Easier to test, debug, and extend individual components

## [0.1.0] - 2025-11-19

### Added
- Initial release of VisProbe framework
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

[0.1.0]: https://github.com/bilgedemirkaya/VisProbe/releases/tag/v0.1.0
