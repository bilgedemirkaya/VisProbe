"""
Preset configurations for VisProbe robustness testing.

Presets provide curated, validated bundles of perturbations for common use cases.
Each preset includes pre-validated parameter ranges that preserve label semantics.

Threat Models:
- "natural": Tests robustness to environmental perturbations (no adversary)
- "adversarial": Tests robustness to gradient-based attacks (white-box adversary)
- "realistic_attack": Tests adversarial attacks under suboptimal conditions
- "comprehensive": All of the above combined for complete benchmarking
"""

from typing import Any, Dict, List, Optional, Tuple

# Default normalization stats (ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 normalization stats
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# =============================================================================
# THREAT MODEL DEFINITIONS
# =============================================================================

THREAT_MODELS = {
    "passive": {
        "name": "Passive (No Adversary)",
        "description": "Environmental perturbations without active adversary",
        "examples": ["Weather changes", "Sensor noise", "Compression artifacts"],
    },
    "active": {
        "name": "Active (White-Box Adversary)",
        "description": "Gradient-based attacks with full model access",
        "examples": ["FGSM", "PGD", "DeepFool"],
    },
    "active_environmental": {
        "name": "Active + Environmental",
        "description": "Adversary exploiting suboptimal environmental conditions",
        "examples": ["Low-light + small FGSM", "Blur + tiny PGD"],
    },
}


# =============================================================================
# PRESET DEFINITIONS - THREAT MODEL AWARE
# =============================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # NATURAL PERTURBATIONS (Passive Threat Model)
    # =========================================================================
    "natural": {
        "name": "Natural Robustness Test",
        "description": "Tests robustness to realistic environmental perturbations",
        "threat_model": "passive",
        "threat_model_description": "No adversary - only natural/environmental variations",
        "strategies": [
            # Single natural perturbations
            {"type": "brightness", "min_factor": 0.5, "max_factor": 1.5},
            {"type": "contrast", "min_factor": 0.6, "max_factor": 1.4},
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 3.0},
            {"type": "gaussian_noise", "min_std": 0.0, "max_std": 0.05},
            {"type": "jpeg_compression", "min_quality": 20, "max_quality": 100},
            {"type": "gamma", "min_gamma": 0.6, "max_gamma": 1.4},
            # Compositional natural perturbations (key for deployment)
            {
                "type": "compositional",
                "name": "low_light_blur",
                "description": "Low-light conditions with motion blur (e.g., night driving)",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.7)},
                    {"type": "gaussian_blur", "sigma_range": (1.0, 2.5)},
                ],
            },
            {
                "type": "compositional",
                "name": "compressed_noisy",
                "description": "Heavy compression with sensor noise (e.g., low-bandwidth video)",
                "components": [
                    {"type": "jpeg_compression", "quality_range": (20, 50)},
                    {"type": "gaussian_noise", "std_range": (0.02, 0.04)},
                ],
            },
            {
                "type": "compositional",
                "name": "dim_low_contrast",
                "description": "Dim lighting with reduced contrast (e.g., foggy conditions)",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.6)},
                    {"type": "contrast", "factor_range": (0.6, 0.8)},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 2000,
        "compute_cost": "low",
        "use_cases": [
            "Pre-deployment validation",
            "Environmental robustness assessment",
            "Camera/sensor variation testing",
            "Production monitoring baselines",
        ],
        "estimated_time": "10-15 minutes for 100 images",
    },

    # =========================================================================
    # ADVERSARIAL ATTACKS (Active Threat Model)
    # =========================================================================
    "adversarial": {
        "name": "Adversarial Robustness Test",
        "description": "Tests robustness to gradient-based adversarial attacks",
        "threat_model": "active",
        "threat_model_description": "White-box adversary with full gradient access",
        "strategies": [
            # FGSM - Fast single-step attack
            {
                "type": "fgsm",
                "min_eps": 0.0,
                "max_eps": 8 / 255,  # Standard ImageNet epsilon
                "description": "Fast Gradient Sign Method - single step attack",
            },
            # PGD - Stronger iterative attack
            {
                "type": "pgd",
                "eps": 8 / 255,
                "eps_step": 2 / 255,
                "max_iter": 20,
                "min_eps": 0.0,
                "max_eps": 8 / 255,
                "description": "Projected Gradient Descent - strong iterative attack",
            },
            # BIM - Basic Iterative Method
            {
                "type": "bim",
                "eps": 4 / 255,
                "max_iter": 10,
                "min_eps": 0.0,
                "max_eps": 8 / 255,
                "description": "Basic Iterative Method - iterative FGSM",
            },
            # Small epsilon FGSM (often missed vulnerabilities)
            {
                "type": "fgsm",
                "min_eps": 0.0,
                "max_eps": 4 / 255,
                "name": "fgsm_small",
                "description": "Small epsilon FGSM - imperceptible attacks",
            },
        ],
        "property": "label_constant",
        "search_budget": 1500,
        "compute_cost": "high",
        "use_cases": [
            "Security testing",
            "Adversarial ML research",
            "Model hardening validation",
            "Robustness certification",
        ],
        "estimated_time": "15-25 minutes for 100 images (requires ART)",
        "requires": ["adversarial-robustness-toolbox"],
    },

    # =========================================================================
    # REALISTIC ATTACKS (Active + Environmental Threat Model)
    # KEY CONTRIBUTION: What standard tests miss!
    # =========================================================================
    "realistic_attack": {
        "name": "Realistic Attack Scenarios",
        "description": "Adversarial attacks exploiting suboptimal environmental conditions",
        "threat_model": "active_environmental",
        "threat_model_description": "Adversary waits for poor conditions, then uses small perturbations",
        "novelty": (
            "Standard robustness tests check natural and adversarial separately. "
            "Real attackers are smarter: they wait for low-light, blur, or compression, "
            "then use SMALLER adversarial perturbations that would fail on clean images. "
            "A model robust to epsilon=8/255 FGSM on clean images may fail at epsilon=2/255 "
            "in low-light. This preset tests that critical blind spot."
        ),
        "strategies": [
            # LOW-LIGHT + FGSM: Attacker waits for dim conditions
            {
                "type": "compositional",
                "name": "lowlight_fgsm",
                "description": "Low-light conditions + small FGSM attack",
                "threat_scenario": "Attacker waits for dusk/night, uses tiny perturbation",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.7)},
                    {"type": "fgsm", "eps_range": (0.0, 4 / 255)},
                ],
            },
            # MOTION BLUR + PGD: Moving camera/subject exploitation
            {
                "type": "compositional",
                "name": "blur_pgd",
                "description": "Motion blur + small PGD attack",
                "threat_scenario": "Attacker targets fast-moving camera with tiny perturbation",
                "components": [
                    {"type": "gaussian_blur", "sigma_range": (1.5, 3.0)},
                    {"type": "pgd", "eps": 2 / 255, "max_iter": 10},
                ],
            },
            # COMPRESSION + FGSM: Network transmission exploitation
            {
                "type": "compositional",
                "name": "compressed_fgsm",
                "description": "Heavy JPEG compression + small FGSM",
                "threat_scenario": "Attacker exploits lossy video/image transmission",
                "components": [
                    {"type": "jpeg_compression", "quality_range": (30, 50)},
                    {"type": "fgsm", "eps_range": (0.0, 4 / 255)},
                ],
            },
            # TRIPLE THREAT: Multiple conditions + tiny perturbation
            {
                "type": "compositional",
                "name": "triple_threat",
                "description": "Low-light + noise + tiny FGSM - the realistic nightmare",
                "threat_scenario": "Worst-case: poor conditions + imperceptible attack",
                "components": [
                    {"type": "brightness", "factor_range": (0.5, 0.7)},
                    {"type": "gaussian_noise", "std_range": (0.01, 0.03)},
                    {"type": "fgsm", "eps_range": (0.0, 2 / 255)},
                ],
            },
            # CONTRAST + BIM: Reduced visibility exploitation
            {
                "type": "compositional",
                "name": "lowcontrast_bim",
                "description": "Low contrast conditions + iterative attack",
                "threat_scenario": "Attacker exploits hazy/foggy conditions",
                "components": [
                    {"type": "contrast", "factor_range": (0.5, 0.7)},
                    {"type": "bim", "eps": 3 / 255, "max_iter": 5},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 2500,
        "compute_cost": "high",
        "use_cases": [
            "Security-critical deployments",
            "Autonomous vehicle testing",
            "Surveillance system hardening",
            "Real-world attack simulation",
        ],
        "estimated_time": "20-30 minutes for 100 images",
        "requires": ["adversarial-robustness-toolbox"],
        "critical_insight": (
            "If realistic_attack score << min(natural, adversarial), "
            "the model has a critical blind spot to opportunistic attacks."
        ),
    },

    # =========================================================================
    # COMPREHENSIVE: ALL THREAT MODELS
    # =========================================================================
    "comprehensive": {
        "name": "Comprehensive Robustness Benchmark",
        "description": "Complete robustness evaluation across all threat models",
        "threat_model": "all",
        "threat_model_description": "Tests passive, active, and combined threat models",
        "strategies": [
            # --- NATURAL (Passive) ---
            {"type": "brightness", "min_factor": 0.5, "max_factor": 1.5, "category": "natural"},
            {"type": "contrast", "min_factor": 0.6, "max_factor": 1.4, "category": "natural"},
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 3.0, "category": "natural"},
            {"type": "gaussian_noise", "min_std": 0.0, "max_std": 0.05, "category": "natural"},
            {"type": "jpeg_compression", "min_quality": 20, "max_quality": 100, "category": "natural"},
            {
                "type": "compositional",
                "name": "low_light_blur",
                "category": "natural",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.7)},
                    {"type": "gaussian_blur", "sigma_range": (1.0, 2.0)},
                ],
            },
            # --- ADVERSARIAL (Active) ---
            {
                "type": "fgsm",
                "min_eps": 0.0,
                "max_eps": 8 / 255,
                "category": "adversarial",
            },
            {
                "type": "pgd",
                "eps": 8 / 255,
                "max_iter": 20,
                "min_eps": 0.0,
                "max_eps": 8 / 255,
                "category": "adversarial",
            },
            # --- REALISTIC ATTACK (Active + Environmental) ---
            {
                "type": "compositional",
                "name": "lowlight_fgsm",
                "category": "realistic_attack",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.7)},
                    {"type": "fgsm", "eps_range": (0.0, 4 / 255)},
                ],
            },
            {
                "type": "compositional",
                "name": "blur_pgd",
                "category": "realistic_attack",
                "components": [
                    {"type": "gaussian_blur", "sigma_range": (1.5, 3.0)},
                    {"type": "pgd", "eps": 2 / 255, "max_iter": 10},
                ],
            },
            {
                "type": "compositional",
                "name": "compressed_fgsm",
                "category": "realistic_attack",
                "components": [
                    {"type": "jpeg_compression", "quality_range": (30, 50)},
                    {"type": "fgsm", "eps_range": (0.0, 4 / 255)},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 5000,
        "compute_cost": "very_high",
        "use_cases": [
            "Research benchmarking",
            "Complete model evaluation",
            "Publication-ready results",
            "Comparing architectures",
        ],
        "estimated_time": "45-60 minutes for 100 images",
        "requires": ["adversarial-robustness-toolbox"],
        "outputs_threat_breakdown": True,
    },

    # =========================================================================
    # LEGACY PRESETS (Backward Compatibility)
    # =========================================================================
    "standard": {
        "name": "Standard Robustness Test (Legacy)",
        "description": "Balanced mix of natural perturbations including compositional failures",
        "threat_model": "passive",
        "threat_model_description": "Legacy preset - equivalent to 'natural' with fewer strategies",
        "strategies": [
            {"type": "brightness", "min_factor": 0.6, "max_factor": 1.4},
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 2.5},
            {"type": "gaussian_noise", "min_std": 0.0, "max_std": 0.03},
            {"type": "jpeg_compression", "min_quality": 40, "max_quality": 100},
            {
                "type": "compositional",
                "name": "low_light_blur",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.7)},
                    {"type": "gaussian_blur", "sigma_range": (1.0, 2.0)},
                ],
            },
            {
                "type": "compositional",
                "name": "compressed_noisy",
                "components": [
                    {"type": "jpeg_compression", "quality_range": (20, 50)},
                    {"type": "gaussian_noise", "std_range": (0.02, 0.05)},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 2000,
        "compute_cost": "low",
        "use_cases": [
            "General robustness testing",
            "Pre-deployment validation",
            "Comparing model architectures",
        ],
        "estimated_time": "10-15 minutes for 100 images",
        "_legacy": True,
        "_migration_hint": "Consider using 'natural' for similar results or 'comprehensive' for complete testing",
    },

    "lighting": {
        "name": "Lighting Conditions (Legacy)",
        "description": "Tests robustness to brightness, contrast, and gamma variations",
        "threat_model": "passive",
        "threat_model_description": "Legacy lighting-focused subset of natural perturbations",
        "strategies": [
            {"type": "brightness", "min_factor": 0.5, "max_factor": 1.5},
            {"type": "contrast", "min_factor": 0.7, "max_factor": 1.3},
            {"type": "gamma", "min_gamma": 0.7, "max_gamma": 1.3},
            {
                "type": "compositional",
                "name": "dim_low_contrast",
                "components": [
                    {"type": "brightness", "factor_range": (0.4, 0.6)},
                    {"type": "contrast", "factor_range": (0.7, 0.9)},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 1000,
        "compute_cost": "low",
        "use_cases": [
            "Outdoor cameras (varying daylight)",
            "Time-of-day robustness",
            "Low-light performance",
        ],
        "estimated_time": "5-8 minutes for 100 images",
        "_legacy": True,
        "_migration_hint": "Consider using 'natural' for broader coverage",
    },

    "blur": {
        "name": "Blur & Defocus (Legacy)",
        "description": "Tests robustness to various types of blur and compression artifacts",
        "threat_model": "passive",
        "threat_model_description": "Legacy blur-focused subset of natural perturbations",
        "strategies": [
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 3.0},
            {"type": "motion_blur", "min_kernel": 1, "max_kernel": 25, "angle": 0},
            {"type": "jpeg_compression", "min_quality": 30, "max_quality": 100},
            {
                "type": "compositional",
                "name": "motion_compressed",
                "components": [
                    {"type": "motion_blur", "kernel_range": (10, 20)},
                    {"type": "jpeg_compression", "quality_range": (40, 60)},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 1200,
        "compute_cost": "low",
        "use_cases": [
            "Motion/camera shake",
            "Out-of-focus images",
            "Video frame compression",
        ],
        "estimated_time": "6-10 minutes for 100 images",
        "_legacy": True,
        "_migration_hint": "Consider using 'natural' for broader coverage",
    },

    "corruption": {
        "name": "Image Corruption (Legacy)",
        "description": "Tests robustness to noise, compression artifacts, and degradation",
        "threat_model": "passive",
        "threat_model_description": "Legacy corruption-focused subset of natural perturbations",
        "strategies": [
            {"type": "gaussian_noise", "min_std": 0.0, "max_std": 0.05},
            {"type": "jpeg_compression", "min_quality": 10, "max_quality": 100},
            {"type": "gaussian_blur", "kernel_size": 5, "min_sigma": 0.0, "max_sigma": 2.0},
            {
                "type": "compositional",
                "name": "degraded_transmission",
                "components": [
                    {"type": "jpeg_compression", "quality_range": (10, 30)},
                    {"type": "gaussian_noise", "std_range": (0.03, 0.05)},
                ],
            },
        ],
        "property": "label_constant",
        "search_budget": 1200,
        "compute_cost": "low",
        "use_cases": [
            "Lossy transmission",
            "Low-bandwidth scenarios",
            "Noisy sensors",
        ],
        "estimated_time": "6-10 minutes for 100 images",
        "_legacy": True,
        "_migration_hint": "Consider using 'natural' for broader coverage",
    },
}


# =============================================================================
# PRESET CATEGORIES
# =============================================================================

PRESET_CATEGORIES = {
    "threat_aware": ["natural", "adversarial", "realistic_attack", "comprehensive"],
    "legacy": ["standard", "lighting", "blur", "corruption"],
    "requires_art": ["adversarial", "realistic_attack", "comprehensive"],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name

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

    preset = PRESETS[name].copy()

    # Warn about legacy presets
    if preset.get("_legacy"):
        hint = preset.get("_migration_hint", "")
        import warnings
        warnings.warn(
            f"Preset '{name}' is legacy. {hint}",
            DeprecationWarning,
            stacklevel=2,
        )

    return preset


def get_preset_by_threat_model(threat_model: str) -> Optional[str]:
    """
    Get the recommended preset for a threat model.

    Args:
        threat_model: One of "passive", "active", "active_environmental", "all"

    Returns:
        Preset name or None if not found
    """
    mapping = {
        "passive": "natural",
        "active": "adversarial",
        "active_environmental": "realistic_attack",
        "all": "comprehensive",
    }
    return mapping.get(threat_model)


def list_presets(include_legacy: bool = False) -> List[Tuple[str, str, str]]:
    """
    List all available presets with descriptions and threat models.

    Args:
        include_legacy: If True, include legacy presets

    Returns:
        List of (name, description, threat_model) tuples
    """
    results = []
    for name, config in PRESETS.items():
        if not include_legacy and config.get("_legacy"):
            continue
        results.append((
            name,
            config["description"],
            config.get("threat_model", "unknown"),
        ))
    return results


def list_threat_aware_presets() -> List[Tuple[str, str]]:
    """
    List only the new threat-model-aware presets.

    Returns:
        List of (name, description) tuples for non-legacy presets
    """
    return [
        (name, PRESETS[name]["description"])
        for name in PRESET_CATEGORIES["threat_aware"]
    ]


def get_preset_info(name: str) -> str:
    """
    Get detailed information about a preset.

    Args:
        name: Preset name

    Returns:
        Formatted string with preset details
    """
    preset = get_preset(name)

    lines = [
        f"Preset: {preset['name']}",
        f"Description: {preset['description']}",
        "",
        f"Threat Model: {preset.get('threat_model', 'unknown')}",
        f"  {preset.get('threat_model_description', '')}",
        "",
    ]

    # Add novelty for realistic_attack
    if "novelty" in preset:
        lines.extend([
            "Key Insight:",
            f"  {preset['novelty']}",
            "",
        ])

    lines.extend([
        f"Compute Cost: {preset.get('compute_cost', 'unknown')}",
        f"Estimated Time: {preset.get('estimated_time', 'unknown')}",
        f"Search Budget: {preset.get('search_budget', 'N/A')} queries",
        "",
        "Use Cases:",
    ])

    for use_case in preset.get('use_cases', []):
        lines.append(f"  - {use_case}")

    lines.extend([
        "",
        f"Strategies: {len(preset['strategies'])} perturbations",
    ])

    # List strategy types
    strategy_types = []
    for strat in preset['strategies']:
        if strat['type'] == 'compositional':
            strategy_types.append(f"  - {strat.get('name', 'compositional')} (compositional)")
        else:
            strategy_types.append(f"  - {strat['type']}")

    lines.extend(strategy_types[:10])  # Show first 10
    if len(strategy_types) > 10:
        lines.append(f"  ... and {len(strategy_types) - 10} more")

    # Dependencies
    if "requires" in preset:
        lines.extend([
            "",
            "Dependencies:",
        ])
        for dep in preset["requires"]:
            lines.append(f"  - {dep}")

    return "\n".join(lines)


def get_strategies_by_category(preset_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get strategies from a preset grouped by threat category.

    Useful for comprehensive preset to compute per-threat-model scores.

    Args:
        preset_name: Name of preset (typically "comprehensive")

    Returns:
        Dict mapping category names to strategy configs
    """
    preset = get_preset(preset_name)
    categorized = {
        "natural": [],
        "adversarial": [],
        "realistic_attack": [],
        "uncategorized": [],
    }

    for strat in preset["strategies"]:
        category = strat.get("category", "uncategorized")
        if category in categorized:
            categorized[category].append(strat)
        else:
            categorized["uncategorized"].append(strat)

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


def is_adversarial_preset(preset_name: str) -> bool:
    """
    Check if a preset requires adversarial attack capabilities.

    Args:
        preset_name: Name of preset

    Returns:
        True if the preset uses adversarial strategies
    """
    return preset_name in PRESET_CATEGORIES.get("requires_art", [])


def get_threat_model_info(threat_model: str) -> Dict[str, Any]:
    """
    Get information about a threat model.

    Args:
        threat_model: One of "passive", "active", "active_environmental"

    Returns:
        Dict with name, description, and examples
    """
    return THREAT_MODELS.get(threat_model, {
        "name": "Unknown",
        "description": "Unknown threat model",
        "examples": [],
    })


# =============================================================================
# VALIDATION STATUS TRACKING
# =============================================================================

VALIDATION_STATUS: Dict[str, Dict[str, Any]] = {
    "natural": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Pending manual validation with 50+ images",
    },
    "adversarial": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Requires ART installation for validation",
    },
    "realistic_attack": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Requires ART installation for validation",
    },
    "comprehensive": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Requires ART installation for validation",
    },
    # Legacy presets
    "standard": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Legacy preset - pending validation",
    },
    "lighting": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Legacy preset - pending validation",
    },
    "blur": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Legacy preset - pending validation",
    },
    "corruption": {
        "validated": False,
        "validation_date": None,
        "label_preservation_rate": None,
        "notes": "Legacy preset - pending validation",
    },
}
