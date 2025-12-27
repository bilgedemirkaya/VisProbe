"""
Single-strategy threshold finding for VisProbe.

Provides the `search()` function for finding exact failure thresholds
for a single perturbation strategy.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .report import Report
from .core.search_engine import SearchEngine
from .properties.classification import LabelConstant
from .strategies.base import Strategy


# Type aliases
ModelLike = nn.Module
DataLike = Union[DataLoader, TensorDataset, List[tuple], torch.Tensor]


def _auto_detect_device() -> torch.device:
    """Auto-detect the best available device."""
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
        for item in data:
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
            samples.append((img.to(device), 0))
        return samples

    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected DataLoader, TensorDataset, list of tuples, or tensor."
        )


def search(
    model: ModelLike,
    data: DataLike,
    strategy: Union[Strategy, Callable[[float], Strategy]],
    property_fn: Optional[Callable] = None,
    mode: str = "adaptive",
    level_lo: float = 0.0,
    level_hi: float = 1.0,
    initial_level: Optional[float] = None,
    step: Optional[float] = None,
    min_step: float = 0.001,
    max_queries: int = 100,
    pass_threshold: float = 0.9,
    device: str = "auto",
    verbose: bool = True,
) -> Report:
    """
    Find exact failure threshold for a single perturbation strategy.

    Use this to:
    - Deep-dive on a specific attack (e.g., "What's the exact FGSM epsilon threshold?")
    - Compare different strategies (e.g., FGSM vs PGD thresholds)
    - Test custom compositions (e.g., "How does low-light affect adversarial threshold?")

    For testing multiple strategies at once, use quick_check() instead.

    Args:
        model: PyTorch model to test
        data: Test data (DataLoader, list of (img, label), or tensor)
        strategy: Perturbation strategy - can be:
            - Fixed strategy: BrightnessStrategy(brightness_factor=0.5)
            - Factory function: lambda level: FGSMStrategy(eps=level)
        property_fn: Property to test (default: LabelConstant - label shouldn't change)
        mode: Search algorithm - "adaptive" (step-halving) or "binary"
        level_lo: Lower bound for search range
        level_hi: Upper bound for search range
        initial_level: Starting level (adaptive mode only, defaults to level_lo)
        step: Initial step size (adaptive mode only, defaults to (level_hi - level_lo) / 10)
        min_step: Minimum step before stopping
        max_queries: Maximum search iterations
        pass_threshold: Fraction of samples that must pass (default 0.9 = 90%)
        device: Device ("auto", "cuda", "cpu", "mps")
        verbose: Print progress

    Returns:
        Report with:
        - failure_threshold: Exact level where model starts failing
        - score: Robustness score (0-1)
        - failures: List of failure cases
        - search_path: Search history

    Example:
        >>> from visprobe import search
        >>> from visprobe.strategies.adversarial import FGSMStrategy
        >>>
        >>> # Find exact FGSM threshold
        >>> report = search(
        ...     model=my_model,
        ...     data=test_data,
        ...     strategy=lambda level: FGSMStrategy(eps=level),
        ...     level_lo=0.0,
        ...     level_hi=16/255,
        ...     mode="binary"
        ... )
        >>> print(f"Model fails at eps >= {report.failure_threshold:.4f}")
    """
    start_time = time.time()

    # 1. Auto-detect device
    if device == "auto":
        device_obj = _auto_detect_device()
    else:
        device_obj = torch.device(device)

    if verbose:
        print(f"Using device: {device_obj}")

    # 2. Prepare model
    model = model.to(device_obj)
    model.eval()

    # 3. Normalize data
    if verbose:
        print("Preparing data...")
    samples = _normalize_data(data, device_obj)
    if verbose:
        print(f"   Testing on {len(samples)} samples")

    # 4. Default property: label shouldn't change
    if property_fn is None:
        property_fn = LabelConstant()

    # 5. Convert strategy to factory if needed
    if callable(strategy) and not isinstance(strategy, Strategy):
        # Already a factory function
        strategy_factory = strategy
        strategy_name = "custom_strategy"
    else:
        # Fixed strategy - wrap in lambda that ignores level
        fixed_strategy = strategy
        strategy_factory = lambda level: fixed_strategy
        strategy_name = strategy.__class__.__name__

    # 6. Create SearchEngine
    engine = SearchEngine(
        model=model,
        strategy_factory=strategy_factory,
        property_fn=property_fn,
        samples=samples,
        mode=mode,
        level_lo=level_lo,
        level_hi=level_hi,
        initial_level=initial_level,
        step=step,
        min_step=min_step,
        max_queries=max_queries,
        pass_threshold=pass_threshold,
        device=device_obj,
    )

    # 7. Run search with progress bar
    if verbose:
        print(f"\nSearching for failure threshold (mode: {mode})...")
        with tqdm(total=max_queries, desc="Searching", leave=False) as pbar:
            result = engine.run(progress_bar=pbar)
    else:
        result = engine.run()

    runtime = time.time() - start_time

    # 8. Print results
    if verbose:
        print(f"\nSearch complete!")
        print(f"   Failure threshold: {result['failure_threshold']:.4f}")
        print(f"   Last pass level: {result['last_pass_level']:.4f}")
        print(f"   Robustness score: {result['robustness_score']:.2%}")
        print(f"   Failures found: {len(result['failures'])}")
        print(f"   Queries used: {result['queries']}")
        print(f"   Runtime: {runtime:.1f}s")

    # 9. Build and return Report
    # Calculate unique failed samples
    unique_failed_indices = set()
    for failure in result["failures"]:
        unique_failed_indices.add(failure["index"])

    passed_samples = len(samples) - len(unique_failed_indices)

    report = Report(
        test_name=f"search_{strategy_name}",
        test_type="search",
        runtime=runtime,
        model_queries=result["queries"] * len(samples),  # Total forward passes
        model_name=model.__class__.__name__,
        preset=None,
        dataset=f"{len(samples)} samples",
        property_name="LabelConstant",
        strategy=strategy_name,
        metrics={
            "overall_robustness_score": result["robustness_score"],
            "failure_threshold": result["failure_threshold"],
            "last_pass_level": result["last_pass_level"],
            "total_failures": len(result["failures"]),
            "unique_failed_samples": len(unique_failed_indices),
        },
        search={
            "mode": mode,
            "level_lo": level_lo,
            "level_hi": level_hi,
            "max_queries": max_queries,
            "pass_threshold": pass_threshold,
            "search_path": result["search_path"],
            "results": [{
                "strategy": strategy_name,
                "strategy_type": "custom",
                "robustness_score": result["robustness_score"],
                "failure_threshold": result["failure_threshold"],
                "failures": result["failures"],
            }],
        },
        total_samples=len(samples),
        passed_samples=passed_samples,
        failure_threshold=result["failure_threshold"],
        search_path=result["search_path"],
    )

    return report
