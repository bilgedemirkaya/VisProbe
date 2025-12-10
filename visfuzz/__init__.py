"""
visfuzz - Robustness testing utilities for vision models.

This package provides:
- perturbations: GPU-optimized image perturbations (noise, blur, lighting, spatial)
- search: Algorithms for finding robustness thresholds
- report: Test result reporting and visualization (from visprobe.api.report)
"""

from .perturbations import (
    # Base
    Perturbation,
    BatchedPerturbation,
    # Noise
    GaussianNoise,
    SaltPepperNoise,
    UniformNoise,
    SpeckleNoise,
    # Blur
    GaussianBlur,
    MotionBlur,
    DefocusBlur,
    BoxBlur,
    # Lighting
    Brightness,
    Contrast,
    Gamma,
    LowLight,
    Saturation,
    # Spatial
    Rotation,
    Scale,
    Translation,
    Shear,
    ElasticDeform,
    # Composition
    Compose,
    RandomChoice,
    Blend,
    LowLightBlur,
    # Registry
    get_perturbation,
    PERTURBATION_REGISTRY,
)

from .search import (
    SearchResult,
    SearchConfig,
    binary_search,
    adaptive_search,
    grid_search,
    label_preserved,
    confidence_above,
    top_k_preserved,
)

# Re-export Report and Failure from visprobe.api.report
try:
    from src.visprobe.api.report import Report, Failure
except ImportError:
    # Fallback if installed as package
    try:
        from visprobe.api.report import Report, Failure
    except ImportError:
        Report = None
        Failure = None

__all__ = [
    # Base
    "Perturbation",
    "BatchedPerturbation",
    # Noise
    "GaussianNoise",
    "SaltPepperNoise",
    "UniformNoise",
    "SpeckleNoise",
    # Blur
    "GaussianBlur",
    "MotionBlur",
    "DefocusBlur",
    "BoxBlur",
    # Lighting
    "Brightness",
    "Contrast",
    "Gamma",
    "LowLight",
    "Saturation",
    # Spatial
    "Rotation",
    "Scale",
    "Translation",
    "Shear",
    "ElasticDeform",
    # Composition
    "Compose",
    "RandomChoice",
    "Blend",
    "LowLightBlur",
    # Registry
    "get_perturbation",
    "PERTURBATION_REGISTRY",
    # Search
    "SearchResult",
    "SearchConfig",
    "binary_search",
    "adaptive_search",
    "grid_search",
    "label_preserved",
    "confidence_above",
    "top_k_preserved",
    # Report
    "Report",
    "Failure",
]

__version__ = "0.1.0"
