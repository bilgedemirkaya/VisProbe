"""
Simplified one-liner API for VisProbe robustness testing.

Provides `quick_check()` for easy-to-use robustness testing with minimal configuration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .api.report import Report
from .presets import PRESETS, get_preset
from .properties.classification import LabelConstant
from .strategies.base import CompositeStrategy, Strategy

logger = logging.getLogger(__name__)

# Type aliases
ModelLike = nn.Module
DataLike = Union[DataLoader, TensorDataset, List[tuple], torch.Tensor]


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

    # Handle individual strategies
    if strategy_type == "brightness":
        # For adaptive search, we just need one instance; level will be varied
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

    elif strategy_type == "compositional":
        # For compositional, use the first component's bounds
        # (This is simplified; could be more sophisticated)
        first_comp = config["components"][0]
        return _extract_level_bounds(first_comp)

    else:
        return (0.0, 1.0)  # Default


def _simple_adaptive_search(
    model: nn.Module,
    strategy: Strategy,
    samples: List[tuple],
    level_min: float,
    level_max: float,
    property_fn: Any,
    max_queries: int = 100,
    progress_bar: Optional[tqdm] = None,
) -> Dict[str, Any]:
    """
    Simple adaptive search to find failure threshold for a strategy.

    Args:
        model: Model to test
        strategy: Perturbation strategy
        samples: List of (image, label) tuples
        level_min: Minimum perturbation level
        level_max: Maximum perturbation level
        property_fn: Property function to evaluate
        max_queries: Maximum number of search iterations
        progress_bar: Optional tqdm progress bar

    Returns:
        Dictionary with search results
    """
    model.eval()

    # Start with level at min (no perturbation)
    current_level = level_min
    step_size = (level_max - level_min) / 10.0  # Initial step
    queries = 0
    failures = []

    last_pass_level = level_min
    first_fail_level = level_max

    # Search for failure threshold
    while queries < max_queries and step_size > 0.001:
        queries += 1

        # Test all samples at current level
        passed_count = 0
        failed_samples = []

        for idx, (img, label) in enumerate(samples):
            # Get original prediction
            with torch.no_grad():
                orig_out = model(img.unsqueeze(0))
                orig_pred = torch.argmax(orig_out, dim=-1).item()

            # Apply perturbation
            perturbed = strategy.generate(img.unsqueeze(0), model, level=current_level)

            # Get perturbed prediction
            with torch.no_grad():
                pert_out = model(perturbed)
                pert_pred = torch.argmax(pert_out, dim=-1).item()

            # Check property (label constant)
            if orig_pred == pert_pred:
                passed_count += 1
            else:
                failed_samples.append({
                    "index": idx,
                    "level": current_level,
                    "original_pred": orig_pred,
                    "perturbed_pred": pert_pred,
                })

        pass_rate = passed_count / len(samples)

        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix({"level": f"{current_level:.3f}", "pass_rate": f"{pass_rate:.2%}"})

        # Adaptive step: if passed, increase level; if failed, decrease
        if pass_rate >= 0.9:  # 90% pass threshold
            last_pass_level = current_level
            current_level += step_size
            if current_level > level_max:
                current_level = level_max
                break
        else:
            first_fail_level = current_level
            failures.extend(failed_samples)
            current_level -= step_size / 2.0
            step_size /= 2.0  # Halve step size

            if current_level < level_min:
                current_level = level_min
                break

    # Calculate robustness score (0-1, higher is better)
    if level_max > level_min:
        score = (last_pass_level - level_min) / (level_max - level_min)
    else:
        score = 1.0

    return {
        "failure_threshold": first_fail_level,
        "last_pass_level": last_pass_level,
        "robustness_score": score,
        "queries": queries,
        "failures": failures[:10],  # Keep top 10
    }


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

    Args:
        model: PyTorch model to test (nn.Module)
        data: Test data (DataLoader, TensorDataset, list of (img, label) tuples, or tensor)
        preset: Preset name ("standard", "lighting", "blur", or "corruption")
        budget: Maximum number of model queries (default: 1000)
        device: Device to use ("auto", "cuda", "cpu", or "mps")
        output_dir: Directory to save results (default: "visprobe_results")
        mean: Optional normalization mean (defaults to ImageNet)
        std: Optional normalization std (defaults to ImageNet)

    Returns:
        Report object with results

    Example:
        >>> from visprobe import quick_check
        >>> import torchvision.models as models
        >>>
        >>> model = models.resnet18(pretrained=True)
        >>> data = ...  # Your test data
        >>>
        >>> report = quick_check(model, data, preset="standard")
        >>> report.show()
        >>> print(f"Robustness score: {report.score}")
    """
    start_time = time.time()

    # 1. Auto-detect device
    device_obj: torch.device
    if device == "auto":
        device_obj = _auto_detect_device()
    else:
        device_obj = torch.device(device) if isinstance(device, str) else device

    print(f"🔧 Using device: {device_obj}")

    # Move model to device
    model = model.to(device_obj)
    model.eval()

    # 2. Load preset configuration
    try:
        preset_config = get_preset(preset)
    except ValueError as e:
        raise ValueError(str(e))

    print(f"📋 Loaded preset: {preset_config['name']}")
    print(f"   {preset_config['description']}")

    # 3. Normalize data
    print(f"📊 Preparing data...")
    samples = _normalize_data(data, device_obj)
    print(f"   Testing on {len(samples)} samples")

    # 4. Set default normalization (ImageNet)
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    # 5. Run search for each strategy in preset
    print(f"\n🔍 Running robustness tests...")
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

        # Run adaptive search with progress bar
        with tqdm(total=queries_per_strategy, desc=f"  {strategy_name}", leave=False) as pbar:
            result = _simple_adaptive_search(
                model=model,
                strategy=strategy,
                samples=samples,
                level_min=level_min,
                level_max=level_max,
                property_fn=property_fn,
                max_queries=queries_per_strategy,
                progress_bar=pbar,
            )

        result["strategy"] = strategy_name
        result["strategy_type"] = strat_config["type"]
        results.append(result)
        total_queries += result["queries"]
        all_failures.extend(result["failures"])

        print(f"    ✓ Failure threshold: {result['failure_threshold']:.3f}")
        print(f"    ✓ Robustness score: {result['robustness_score']:.2%}")

    # 6. Compute overall robustness score
    overall_score = sum(r["robustness_score"] for r in results) / len(results)

    runtime = time.time() - start_time

    # Calculate unique failed samples (not counting duplicates across strategies)
    unique_failed_indices = set()
    for failure in all_failures:
        unique_failed_indices.add(failure["index"])

    passed_samples = len(samples) - len(unique_failed_indices)

    # 7. Build report
    print(f"\n✅ Testing complete!")
    print(f"   Overall robustness score: {overall_score:.2%}")
    print(f"   Total failures found: {len(all_failures)}")
    print(f"   Unique failed samples: {len(unique_failed_indices)}")
    print(f"   Runtime: {runtime:.1f}s")
    print(f"   Model queries: {total_queries}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    report = Report(
        test_name=f"quick_check_{preset}",
        test_type="quick_check",
        runtime=runtime,
        model_queries=total_queries,
        model_name=model.__class__.__name__,
        dataset=f"{len(samples)} samples",
        property_name="LabelConstant",
        strategy=preset,
        metrics={
            "overall_robustness_score": overall_score,
            "total_failures": len(all_failures),
            "unique_failed_samples": len(unique_failed_indices),
            "strategies_tested": len(results),
        },
        search={
            "preset": preset,
            "budget": budget,
            "results": results,
        },
        total_samples=len(samples),
        passed_samples=passed_samples,
    )

    # Save report
    report.save()

    return report
