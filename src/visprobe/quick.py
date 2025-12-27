"""
Simplified one-liner API for VisProbe robustness testing.

Provides `quick_check()` for easy-to-use robustness testing with minimal configuration.
Supports natural perturbations, adversarial attacks, and realistic attack scenarios.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .report import Report
from .core.search_engine import SearchEngine
from .presets import (
    PRESETS,
    PRESET_CATEGORIES,
    get_preset,
    get_strategies_by_category,
    is_adversarial_preset,
)
from .properties.classification import LabelConstant
from .strategies.base import CompositeStrategy, Strategy

logger = logging.getLogger(__name__)

# Type aliases
ModelLike = nn.Module
DataLike = Union[DataLoader, TensorDataset, List[tuple], torch.Tensor]

# Flag for adversarial strategy availability
_ADVERSARIAL_AVAILABLE = False
try:
    from .strategies.adversarial import (
        BIMStrategy,
        FGSMStrategy,
        PGDStrategy,
    )
    _ADVERSARIAL_AVAILABLE = True
except ImportError:
    FGSMStrategy = None
    PGDStrategy = None
    BIMStrategy = None


def _check_adversarial_available(preset_name: str) -> None:
    """Check if adversarial strategies are available for a preset."""
    if is_adversarial_preset(preset_name) and not _ADVERSARIAL_AVAILABLE:
        raise ImportError(
            f"Preset '{preset_name}' requires adversarial strategies.\n"
            "Install the Adversarial Robustness Toolbox:\n"
            "  pip install adversarial-robustness-toolbox"
        )


def _auto_detect_device() -> torch.device:
    """
    Auto-detect the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _normalize_data(
    data: DataLike,
    device: torch.device,
) -> List[tuple]:
    """
    Normalize various data formats into a list of (image, label) tuples.

    Args:
        data: Input data in various formats
        device: Device to move tensors to

    Returns:
        List of (image_tensor, label) tuples
    """
    # Case 1: DataLoader
    if isinstance(data, DataLoader):
        samples = []
        for batch in data:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
            else:
                images = batch
                labels = torch.zeros(images.shape[0], dtype=torch.long)

            for img, lbl in zip(images, labels):
                samples.append((img.to(device), int(lbl.item())))
        return samples

    # Case 2: TensorDataset
    elif isinstance(data, TensorDataset):
        samples = []
        for item in data:  # type: ignore[attr-defined]
            if len(item) == 2:
                img, lbl = item
            else:
                img = item[0]
                lbl = 0
            samples.append((img.to(device), int(lbl)))
        return samples

    # Case 3: List of tuples
    elif isinstance(data, list):
        samples = []
        for item in data:
            if isinstance(item, tuple) and len(item) == 2:
                img, lbl = item
                samples.append((img.to(device), int(lbl)))
            else:
                raise ValueError(
                    f"List items must be (image, label) tuples, got {type(item)}"
                )
        return samples

    # Case 4: Single tensor (batch of images)
    elif isinstance(data, torch.Tensor):
        samples = []
        for img in data:
            samples.append((img.to(device), 0))  # Default label
        return samples

    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected DataLoader, TensorDataset, list of tuples, or tensor."
        )


def _instantiate_strategy_from_config(
    config: Dict[str, Any],
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None,
) -> Strategy:
    """
    Instantiate a Strategy from a preset configuration dict.

    Args:
        config: Strategy configuration dictionary
        mean: Normalization mean (for noise strategies)
        std: Normalization std (for noise strategies)

    Returns:
        Instantiated Strategy object
    """
    from .strategies.image import (
        BrightnessStrategy,
        ContrastStrategy,
        GammaStrategy,
        GaussianBlurStrategy,
        GaussianNoiseStrategy,
        JPEGCompressionStrategy,
        MotionBlurStrategy,
    )

    strategy_type = config["type"]

    # Handle compositional strategies
    if strategy_type == "compositional":
        components = []
        for comp_config in config["components"]:
            comp_strategy = _instantiate_strategy_from_config(comp_config, mean, std)
            components.append(comp_strategy)
        return CompositeStrategy(components)

    # =========================================================================
    # NATURAL PERTURBATION STRATEGIES
    # =========================================================================
    if strategy_type == "brightness":
        return BrightnessStrategy(brightness_factor=1.0)

    elif strategy_type == "contrast":
        return ContrastStrategy(contrast_factor=1.0)

    elif strategy_type == "gamma":
        return GammaStrategy(gamma=1.0)

    elif strategy_type == "gaussian_blur":
        kernel_size = config.get("kernel_size", 5)
        return GaussianBlurStrategy(kernel_size=kernel_size, sigma=0.0)

    elif strategy_type == "motion_blur":
        angle = config.get("angle", 0.0)
        return MotionBlurStrategy(kernel_size=1, angle=angle)

    elif strategy_type == "jpeg_compression":
        return JPEGCompressionStrategy(quality=100)

    elif strategy_type == "gaussian_noise":
        return GaussianNoiseStrategy(std_dev=0.0, mean=mean, std=std)

    # =========================================================================
    # ADVERSARIAL ATTACK STRATEGIES
    # =========================================================================
    elif strategy_type == "fgsm":
        if not _ADVERSARIAL_AVAILABLE:
            raise ImportError(
                "FGSM strategy requires adversarial-robustness-toolbox. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        eps = config.get("eps", config.get("max_eps", 8 / 255))
        return FGSMStrategy(eps=eps)

    elif strategy_type == "pgd":
        if not _ADVERSARIAL_AVAILABLE:
            raise ImportError(
                "PGD strategy requires adversarial-robustness-toolbox. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        eps = config.get("eps", config.get("max_eps", 8 / 255))
        eps_step = config.get("eps_step", eps / 10)
        max_iter = config.get("max_iter", 20)
        return PGDStrategy(eps=eps, eps_step=eps_step, max_iter=max_iter)

    elif strategy_type == "bim":
        if not _ADVERSARIAL_AVAILABLE:
            raise ImportError(
                "BIM strategy requires adversarial-robustness-toolbox. "
                "Install with: pip install adversarial-robustness-toolbox"
            )
        eps = config.get("eps", config.get("max_eps", 4 / 255))
        max_iter = config.get("max_iter", 10)
        return BIMStrategy(eps=eps, max_iter=max_iter)

    else:
        raise ValueError(f"Unknown strategy type in preset: {strategy_type}")


def _extract_level_bounds(config: Dict[str, Any]) -> tuple[float, float]:
    """
    Extract (min_level, max_level) from a strategy config.

    Args:
        config: Strategy configuration dictionary

    Returns:
        Tuple of (min_level, max_level)
    """
    strategy_type = config["type"]

    # Natural perturbations
    if strategy_type == "brightness":
        return (config.get("min_factor", 0.5), config.get("max_factor", 1.5))

    elif strategy_type == "contrast":
        return (config.get("min_factor", 0.7), config.get("max_factor", 1.3))

    elif strategy_type == "gamma":
        return (config.get("min_gamma", 0.7), config.get("max_gamma", 1.3))

    elif strategy_type == "gaussian_blur":
        return (config.get("min_sigma", 0.0), config.get("max_sigma", 2.5))

    elif strategy_type == "motion_blur":
        return (config.get("min_kernel", 1), config.get("max_kernel", 25))

    elif strategy_type == "jpeg_compression":
        # Note: for JPEG, LOWER quality = MORE corruption
        # So we'll search from max (no corruption) to min (high corruption)
        return (config.get("min_quality", 10), config.get("max_quality", 100))

    elif strategy_type == "gaussian_noise":
        return (config.get("min_std", 0.0), config.get("max_std", 0.05))

    # Adversarial attacks - use epsilon ranges
    elif strategy_type == "fgsm":
        return (config.get("min_eps", 0.0), config.get("max_eps", 8 / 255))

    elif strategy_type == "pgd":
        return (config.get("min_eps", 0.0), config.get("max_eps", 8 / 255))

    elif strategy_type == "bim":
        return (config.get("min_eps", 0.0), config.get("max_eps", 8 / 255))

    elif strategy_type == "compositional":
        # For compositional, use the first component's bounds
        # (This is simplified; could be more sophisticated)
        first_comp = config["components"][0]
        return _extract_level_bounds(first_comp)

    else:
        return (0.0, 1.0)  # Default


def _is_adversarial_strategy(strategy_type: str) -> bool:
    """Check if a strategy type is adversarial."""
    return strategy_type in {"fgsm", "pgd", "bim", "apgd", "square_attack"}


def _compute_threat_model_scores(
    results: List[Dict[str, Any]],
    preset_config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute scores broken down by threat model category.

    Args:
        results: List of per-strategy results
        preset_config: Preset configuration dictionary

    Returns:
        Dict mapping threat model names to average scores
    """
    # Get strategy configurations to map results to categories
    strategies = preset_config.get("strategies", [])

    # Build mapping from strategy name/type to category
    strategy_categories = {}
    for strat in strategies:
        name = strat.get("name", strat["type"])
        category = strat.get("category", "uncategorized")
        strategy_categories[name] = category

    # Group results by category
    category_scores = {
        "natural": [],
        "adversarial": [],
        "realistic_attack": [],
    }

    for result in results:
        strategy_name = result.get("strategy", "")
        strategy_type = result.get("strategy_type", "")

        # Try to find category
        category = strategy_categories.get(strategy_name)
        if category is None:
            # Fallback: infer from strategy type
            if _is_adversarial_strategy(strategy_type):
                category = "adversarial"
            elif strategy_type == "compositional":
                # Check if any component is adversarial
                # For now, mark as realistic_attack if it was in that category
                category = strategy_categories.get(strategy_name, "natural")
            else:
                category = "natural"

        if category in category_scores:
            category_scores[category].append(result.get("robustness_score", 0))

    # Compute averages
    threat_scores = {}
    for category, scores in category_scores.items():
        if scores:
            threat_scores[category] = sum(scores) / len(scores)

    return threat_scores


def _check_opportunistic_vulnerability(
    threat_scores: Dict[str, float],
    threshold: float = 0.1,
) -> Optional[str]:
    """
    Check if the model is vulnerable to opportunistic attacks.

    This is the KEY INSIGHT: if realistic_attack score is significantly lower
    than both natural and adversarial scores, the model has a blind spot.

    Args:
        threat_scores: Dict of threat model scores
        threshold: Minimum gap to trigger warning (default 0.1 = 10%)

    Returns:
        Warning message if vulnerable, None otherwise
    """
    natural = threat_scores.get("natural")
    adversarial = threat_scores.get("adversarial")
    realistic = threat_scores.get("realistic_attack")

    if realistic is None:
        return None

    # Check if realistic is significantly worse than BOTH natural and adversarial
    min_individual = None
    if natural is not None and adversarial is not None:
        min_individual = min(natural, adversarial)
    elif natural is not None:
        min_individual = natural
    elif adversarial is not None:
        min_individual = adversarial

    if min_individual is not None and (min_individual - realistic) > threshold:
        return (
            f"Model vulnerable to opportunistic attacks!\n"
            f"  Natural robustness:        {natural:.1%}\n"
            f"  Adversarial robustness:    {adversarial:.1%}\n"
            f"  Realistic attack:          {realistic:.1%} (gap: {min_individual - realistic:.1%})\n"
            f"  Implication: Attackers can exploit environmental conditions "
            f"to succeed with smaller perturbations."
        )

    return None


def quick_check(
    model: ModelLike,
    data: DataLike,
    preset: str = "standard",
    budget: int = 1000,
    device: Union[str, torch.device] = "auto",
    output_dir: str = "visprobe_results",
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None,
) -> Report:
    """
    Quick robustness check with a single function call.

    This is the simplified API for VisProbe. It runs adaptive search over multiple
    perturbations defined in a preset configuration and returns a comprehensive report.

    Supports three threat models:
    - "natural": Environmental perturbations (no adversary)
    - "adversarial": Gradient-based attacks (white-box adversary)
    - "realistic_attack": Adversarial attacks under suboptimal conditions

    Args:
        model: PyTorch model to test (nn.Module)
        data: Test data (DataLoader, TensorDataset, list of (img, label) tuples, or tensor)
        preset: Preset name - Threat-aware: "natural", "adversarial", "realistic_attack", "comprehensive"
                Legacy: "standard", "lighting", "blur", "corruption"
        budget: Maximum number of model queries (default: 1000)
        device: Device to use ("auto", "cuda", "cpu", or "mps")
        output_dir: Directory to save results (default: "visprobe_results")
        mean: Optional normalization mean (defaults to ImageNet)
        std: Optional normalization std (defaults to ImageNet)

    Returns:
        Report object with results including:
        - score: Overall robustness score (0-1)
        - threat_model_scores: Per-threat-model breakdown (for comprehensive)
        - failures: List of failure cases
        - threat_model_summary: Analysis of vulnerabilities

    Example:
        >>> from visprobe import quick_check
        >>> import torchvision.models as models
        >>>
        >>> model = models.resnet18(pretrained=True)
        >>> data = ...  # Your test data
        >>>
        >>> # Test natural robustness
        >>> report = quick_check(model, data, preset="natural")
        >>> print(f"Natural robustness: {report.score:.1%}")
        >>>
        >>> # Test adversarial robustness
        >>> report = quick_check(model, data, preset="adversarial")
        >>> print(f"Adversarial robustness: {report.score:.1%}")
        >>>
        >>> # Test realistic attack scenarios (the critical one!)
        >>> report = quick_check(model, data, preset="realistic_attack")
        >>> print(f"Realistic attack: {report.score:.1%}")
        >>>
        >>> # Comprehensive test with threat breakdown
        >>> report = quick_check(model, data, preset="comprehensive")
        >>> print(f"Threat model scores: {report.metrics['threat_model_scores']}")
    """
    start_time = time.time()

    # 1. Auto-detect device
    device_obj: torch.device
    if device == "auto":
        device_obj = _auto_detect_device()
    else:
        device_obj = torch.device(device) if isinstance(device, str) else device

    print(f"Using device: {device_obj}")

    # Move model to device
    model = model.to(device_obj)
    model.eval()

    # 2. Check adversarial availability for preset
    _check_adversarial_available(preset)

    # 3. Load preset configuration
    try:
        preset_config = get_preset(preset)
    except ValueError as e:
        raise ValueError(str(e))

    threat_model = preset_config.get("threat_model", "unknown")
    print(f"Loaded preset: {preset_config['name']}")
    print(f"   {preset_config['description']}")
    print(f"   Threat model: {threat_model}")

    # Show novelty for realistic_attack
    if "novelty" in preset_config:
        print(f"   Key insight: {preset_config['novelty'][:100]}...")

    # 4. Normalize data
    print(f"Preparing data...")
    samples = _normalize_data(data, device_obj)
    print(f"   Testing on {len(samples)} samples")

    # 5. Set default normalization (ImageNet)
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    # 6. Run search for each strategy in preset
    print(f"\nRunning robustness tests...")
    property_fn = LabelConstant()

    results = []
    total_queries = 0
    all_failures = []

    strategies_config = preset_config["strategies"]
    queries_per_strategy = budget // len(strategies_config)

    for strat_config in strategies_config:
        strategy = _instantiate_strategy_from_config(strat_config, mean, std)
        level_min, level_max = _extract_level_bounds(strat_config)

        strategy_name = strat_config.get("name", strat_config["type"])
        print(f"\n  Testing: {strategy_name}")

        # Create SearchEngine with fixed strategy (wrapped in factory)
        engine = SearchEngine(
            model=model,
            strategy_factory=lambda level, s=strategy: s,  # Fixed strategy, ignores level
            property_fn=property_fn,
            samples=samples,
            mode="adaptive",
            level_lo=level_min,
            level_hi=level_max,
            max_queries=queries_per_strategy,
            device=device_obj,
        )

        # Run adaptive search with progress bar
        with tqdm(total=queries_per_strategy, desc=f"  {strategy_name}", leave=False) as pbar:
            result = engine.run(progress_bar=pbar)

        result["strategy"] = strategy_name
        result["strategy_type"] = strat_config["type"]
        result["category"] = strat_config.get("category", "uncategorized")
        results.append(result)
        total_queries += result["queries"]
        all_failures.extend(result["failures"])

        print(f"    Failure threshold: {result['failure_threshold']:.3f}")
        print(f"    Robustness score: {result['robustness_score']:.2%}")

    # 7. Compute overall robustness score
    overall_score = sum(r["robustness_score"] for r in results) / len(results)

    runtime = time.time() - start_time

    # Calculate unique failed samples (not counting duplicates across strategies)
    unique_failed_indices = set()
    for failure in all_failures:
        unique_failed_indices.add(failure["index"])

    passed_samples = len(samples) - len(unique_failed_indices)

    # 8. Compute threat model breakdown for comprehensive preset
    threat_model_scores = {}
    threat_model_summary = {}
    vulnerability_warning = None

    if preset_config.get("outputs_threat_breakdown") or threat_model == "all":
        threat_model_scores = _compute_threat_model_scores(results, preset_config)
        threat_model_summary = {
            "threat_model": threat_model,
            "scores_by_threat": threat_model_scores,
            "overall_score": overall_score,
        }

        # Check for opportunistic vulnerability
        vulnerability_warning = _check_opportunistic_vulnerability(threat_model_scores)
        if vulnerability_warning:
            threat_model_summary["vulnerability_warning"] = vulnerability_warning

    # 9. Build report
    print(f"\nTesting complete!")
    print(f"   Overall robustness score: {overall_score:.2%}")

    if threat_model_scores:
        print(f"\n   Threat Model Breakdown:")
        for tm, score in threat_model_scores.items():
            print(f"      {tm}: {score:.2%}")

    if vulnerability_warning:
        print(f"\n   CRITICAL WARNING:")
        print(f"   {vulnerability_warning}")

    print(f"\n   Total failures found: {len(all_failures)}")
    print(f"   Unique failed samples: {len(unique_failed_indices)}")
    print(f"   Runtime: {runtime:.1f}s")
    print(f"   Model queries: {total_queries}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build metrics dict with threat model info
    metrics = {
        "overall_robustness_score": overall_score,
        "total_failures": len(all_failures),
        "unique_failed_samples": len(unique_failed_indices),
        "strategies_tested": len(results),
        "threat_model": threat_model,
    }

    if threat_model_scores:
        metrics["threat_model_scores"] = threat_model_scores

    if vulnerability_warning:
        metrics["vulnerability_warning"] = vulnerability_warning

    report = Report(
        test_name=f"quick_check_{preset}",
        test_type="quick_check",
        runtime=runtime,
        model_queries=total_queries,
        model_name=model.__class__.__name__,
        preset=preset,
        dataset=f"{len(samples)} samples",
        property_name="LabelConstant",
        strategy=preset,
        metrics=metrics,
        search={
            "preset": preset,
            "budget": budget,
            "threat_model": threat_model,
            "results": results,
        },
        total_samples=len(samples),
        passed_samples=passed_samples,
    )

    # Add threat model summary if available
    if threat_model_summary:
        report.add_metric("threat_model_summary", threat_model_summary)

    # Save report
    report.save()

    return report


def compare_threat_models(
    model: ModelLike,
    data: DataLike,
    budget: int = 1000,
    device: Union[str, torch.device] = "auto",
    mean: Optional[tuple] = None,
    std: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Run all three threat model presets and compare results.

    This is a convenience function that runs natural, adversarial, and realistic_attack
    presets separately and returns a comparison summary.

    Args:
        model: PyTorch model to test
        data: Test data
        budget: Query budget per preset
        device: Device to use
        mean: Normalization mean
        std: Normalization std

    Returns:
        Dictionary with:
        - reports: Dict of preset name -> Report
        - scores: Dict of preset name -> score
        - comparison: Analysis of relative robustness
        - vulnerability_check: Warning if opportunistic attack vulnerability detected
    """
    reports = {}
    scores = {}

    presets_to_run = ["natural", "adversarial", "realistic_attack"]

    for preset_name in presets_to_run:
        try:
            print(f"\n{'='*60}")
            print(f"Running {preset_name.upper()} preset")
            print(f"{'='*60}\n")

            report = quick_check(
                model=model,
                data=data,
                preset=preset_name,
                budget=budget,
                device=device,
                mean=mean,
                std=std,
            )
            reports[preset_name] = report
            scores[preset_name] = report.score
        except ImportError as e:
            print(f"Skipping {preset_name}: {e}")
            continue

    # Generate comparison
    comparison = {
        "scores": scores,
        "ranking": sorted(scores.keys(), key=lambda k: scores.get(k, 0), reverse=True),
    }

    # Check for vulnerability
    vulnerability_check = _check_opportunistic_vulnerability(scores)
    if vulnerability_check:
        comparison["vulnerability_warning"] = vulnerability_check

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for preset_name in presets_to_run:
        if preset_name in scores:
            score = scores[preset_name]
            bar = "" * int(score * 20)
            print(f"  {preset_name:20s} {score:6.1%} {bar}")

    if vulnerability_check:
        print("\n" + vulnerability_check)

    return {
        "reports": reports,
        "scores": scores,
        "comparison": comparison,
        "vulnerability_check": vulnerability_check,
    }
