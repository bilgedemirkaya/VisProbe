"""
Preset configurations for VisProbe robustness testing.

Presets provide curated, validated bundles of perturbations for common use cases.
Each preset includes pre-validated parameter ranges that preserve label semantics.
"""

from typing import Any, Dict, List, Tuple

# Default normalization stats (ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 normalization stats
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


PRESETS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "name": "Standard Robustness Test",
        "description": "Balanced mix of natural perturbations including compositional failures",
        "strategies": [
            # Single perturbations
            {"type": "brightness", "min_factor": 0.6, "max_factor": 1.4},
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 2.5},
            {"type": "gaussian_noise", "min_std": 0.0, "max_std": 0.03},
            {"type": "jpeg_compression", "min_quality": 40, "max_quality": 100},
            # Compositional perturbations (key innovation)
            {"type": "compositional", "name": "low_light_blur", "components": [
                {"type": "brightness", "factor_range": (0.4, 0.7)},  # low-light
                {"type": "gaussian_blur", "sigma_range": (1.0, 2.0)},  # motion blur
            ]},
            {"type": "compositional", "name": "compressed_noisy", "components": [
                {"type": "jpeg_compression", "quality_range": (20, 50)},
                {"type": "gaussian_noise", "std_range": (0.02, 0.05)},
            ]},
        ],
        "property": "label_constant",
        "search_budget": 2000,
        "use_cases": [
            "General robustness testing",
            "Pre-deployment validation",
            "Comparing model architectures",
        ],
        "estimated_time": "10-15 minutes for 100 images",
    },

    "lighting": {
        "name": "Lighting Conditions",
        "description": "Tests robustness to brightness, contrast, and gamma variations",
        "strategies": [
            {"type": "brightness", "min_factor": 0.5, "max_factor": 1.5},
            {"type": "contrast", "min_factor": 0.7, "max_factor": 1.3},
            {"type": "gamma", "min_gamma": 0.7, "max_gamma": 1.3},
            # Compositional: low-light with reduced contrast
            {"type": "compositional", "name": "dim_low_contrast", "components": [
                {"type": "brightness", "factor_range": (0.4, 0.6)},
                {"type": "contrast", "factor_range": (0.7, 0.9)},
            ]},
        ],
        "property": "label_constant",
        "search_budget": 1000,
        "use_cases": [
            "Outdoor cameras (varying daylight)",
            "Time-of-day robustness",
            "Low-light performance",
        ],
        "estimated_time": "5-8 minutes for 100 images",
    },

    "blur": {
        "name": "Blur & Defocus",
        "description": "Tests robustness to various types of blur and compression artifacts",
        "strategies": [
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 3.0},
            {"type": "motion_blur", "min_kernel": 1, "max_kernel": 25, "angle": 0},
            {"type": "jpeg_compression", "min_quality": 30, "max_quality": 100},
            # Compositional: motion blur with compression (video frame artifacts)
            {"type": "compositional", "name": "motion_compressed", "components": [
                {"type": "motion_blur", "kernel_range": (10, 20)},
                {"type": "jpeg_compression", "quality_range": (40, 60)},
            ]},
        ],
        "property": "label_constant",
        "search_budget": 1200,
        "use_cases": [
            "Motion/camera shake",
            "Out-of-focus images",
            "Video frame compression",
        ],
        "estimated_time": "6-10 minutes for 100 images",
    },

    "corruption": {
        "name": "Image Corruption",
        "description": "Tests robustness to noise, compression artifacts, and degradation",
        "strategies": [
            {"type": "gaussian_noise", "min_std": 0.0, "max_std": 0.05},
            {"type": "jpeg_compression", "min_quality": 10, "max_quality": 100},
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 2.0},
            # Compositional: heavy compression + noise (degraded transmission)
            {"type": "compositional", "name": "degraded_transmission", "components": [
                {"type": "jpeg_compression", "quality_range": (10, 30)},
                {"type": "gaussian_noise", "std_range": (0.03, 0.05)},
            ]},
        ],
        "property": "label_constant",
        "search_budget": 1200,
        "use_cases": [
            "Lossy transmission",
            "Low-bandwidth scenarios",
            "Noisy sensors",
        ],
        "estimated_time": "6-10 minutes for 100 images",
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name ("standard", "lighting", "blur", or "corruption")

    Returns:
        Preset configuration dictionary

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}"
        )
    return PRESETS[name].copy()


def list_presets() -> List[Tuple[str, str]]:
    """
    List all available presets with descriptions.

    Returns:
        List of (name, description) tuples
    """
    return [(name, config["description"]) for name, config in PRESETS.items()]


def get_preset_info(name: str) -> str:
    """
    Get detailed information about a preset.

    Args:
        name: Preset name

    Returns:
        Formatted string with preset details
    """
    preset = get_preset(name)
    info = f"Preset: {preset['name']}\n"
    info += f"Description: {preset['description']}\n"
    info += f"Estimated time: {preset['estimated_time']}\n"
    info += f"Use cases:\n"
    for use_case in preset['use_cases']:
        info += f"  - {use_case}\n"
    info += f"Strategies: {len(preset['strategies'])} perturbations\n"
    return info


# Validation status tracking
# This will be populated after manual validation in Task 1.2
VALIDATION_STATUS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Pending manual validation with 50+ images",
    },
    "lighting": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Pending manual validation with 50+ images",
    },
    "blur": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Pending manual validation with 50+ images",
    },
    "corruption": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Pending manual validation with 50+ images",
    },
}
