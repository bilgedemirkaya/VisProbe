"""
Search algorithms for finding robustness thresholds.

Provides adaptive search strategies for finding minimal perturbation
levels that cause model failures. Optimized for efficiency with:
- Golden section search for faster convergence
- Exponential exploration for unknown bounds
- Evaluation caching to avoid redundant computations
- Batch-aware GPU evaluation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .perturbations.base import Perturbation

__all__ = [
    "SearchResult",
    "SearchConfig",
    "binary_search",
    "adaptive_search",
    "golden_section_search",
    "grid_search",
    "label_preserved",
    "confidence_above",
    "top_k_preserved",
]

# Golden ratio for golden section search
PHI = (1 + math.sqrt(5)) / 2
INVPHI = 1 / PHI
INVPHI2 = 1 / (PHI * PHI)


@dataclass
class SearchResult:
    """
    Result of a robustness threshold search.

    Attributes:
        failure_threshold: Minimum level that caused failure (None if none found)
        converged: Whether the search converged to a threshold
        iterations: Number of search iterations performed
        queries: Number of model forward passes used
        path: History of (level, passed) tuples from search
        perturbed_images: Images at failure threshold (if found)
        per_sample_thresholds: Per-image threshold estimates
        cache_hits: Number of cache hits (evaluations avoided)
    """
    failure_threshold: Optional[float] = None
    converged: bool = False
    iterations: int = 0
    queries: int = 0
    path: List[Dict[str, Any]] = field(default_factory=list)
    perturbed_images: Optional[torch.Tensor] = None
    perturbed_output: Optional[torch.Tensor] = None
    per_sample_thresholds: Optional[List[Optional[float]]] = None
    cache_hits: int = 0


@dataclass
class SearchConfig:
    """
    Configuration for search algorithms.

    Attributes:
        min_level: Minimum perturbation level to search
        max_level: Maximum perturbation level to search
        tolerance: Convergence tolerance (search stops when range < tolerance)
        max_iterations: Maximum number of search iterations
        max_queries: Maximum model forward passes
        reduce: How to aggregate per-sample results ('all', 'any', 'frac>=0.5')
        cache_evaluations: Whether to cache and reuse evaluations
        early_stop_no_improvement: Stop if no improvement for N iterations
    """
    min_level: float = 0.0
    max_level: float = 1.0
    tolerance: float = 1e-4
    max_iterations: int = 50
    max_queries: int = 500
    reduce: str = "all"
    cache_evaluations: bool = True
    early_stop_no_improvement: int = 5

    def __post_init__(self) -> None:
        if self.min_level >= self.max_level:
            raise ValueError(f"min_level ({self.min_level}) must be < max_level ({self.max_level})")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")


class EvaluationCache:
    """Cache for model evaluations to avoid redundant computations."""

    def __init__(self, tolerance: float = 1e-6):
        self._cache: Dict[float, Tuple[bool, float, List[bool]]] = {}
        self._tolerance = tolerance
        self.hits = 0
        self.misses = 0

    def _quantize(self, level: float) -> float:
        """Quantize level for cache key."""
        return round(level / self._tolerance) * self._tolerance

    def get(self, level: float) -> Optional[Tuple[bool, float, List[bool]]]:
        """Get cached result if available."""
        key = self._quantize(level)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        return None

    def set(self, level: float, batch_passed: bool, pass_frac: float, passed_mask: List[bool]) -> None:
        """Cache an evaluation result."""
        key = self._quantize(level)
        self._cache[key] = (batch_passed, pass_frac, passed_mask)
        self.misses += 1

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0


def _evaluate_batch(
    model: nn.Module,
    images: torch.Tensor,
    perturbation: Perturbation,
    level: float,
    property_fn: Callable[[torch.Tensor, torch.Tensor], List[bool]],
    clean_outputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[bool]]:
    """
    Evaluate model on perturbed images and check property.

    Returns:
        Tuple of (perturbed_images, perturbed_outputs, per_sample_passed)
    """
    perturbed = perturbation(images, level)

    with torch.no_grad():
        model.eval()
        pert_outputs = model(perturbed)

    passed_mask = property_fn(clean_outputs, pert_outputs)

    return perturbed, pert_outputs, passed_mask


def _reduce_passed(passed_mask: List[bool], reduce: str) -> Tuple[bool, float]:
    """Reduce per-sample pass/fail to batch-level decision."""
    if not passed_mask:
        return False, 0.0

    pass_frac = sum(passed_mask) / len(passed_mask)

    if reduce == "all":
        return all(passed_mask), pass_frac
    elif reduce == "any":
        return any(passed_mask), pass_frac
    elif reduce.startswith("frac>="):
        threshold = float(reduce.split(">=")[1])
        return pass_frac >= threshold, pass_frac
    else:
        return all(passed_mask), pass_frac


def _evaluate_with_cache(
    model: nn.Module,
    images: torch.Tensor,
    perturbation: Perturbation,
    level: float,
    property_fn: Callable[[torch.Tensor, torch.Tensor], List[bool]],
    clean_outputs: torch.Tensor,
    cache: Optional[EvaluationCache],
    reduce: str,
) -> Tuple[bool, float, List[bool], torch.Tensor, torch.Tensor, bool]:
    """
    Evaluate with optional caching.

    Returns:
        (batch_passed, pass_frac, passed_mask, pert_images, pert_outputs, was_cached)
    """
    if cache is not None:
        cached = cache.get(level)
        if cached is not None:
            batch_passed, pass_frac, passed_mask = cached
            # Still need to generate images for result
            perturbed = perturbation(images, level)
            with torch.no_grad():
                model.eval()
                pert_outputs = model(perturbed)
            return batch_passed, pass_frac, passed_mask, perturbed, pert_outputs, True

    pert_images, pert_outputs, passed_mask = _evaluate_batch(
        model, images, perturbation, level, property_fn, clean_outputs
    )
    batch_passed, pass_frac = _reduce_passed(passed_mask, reduce)

    if cache is not None:
        cache.set(level, batch_passed, pass_frac, passed_mask)

    return batch_passed, pass_frac, passed_mask, pert_images, pert_outputs, False


def golden_section_search(
    model: nn.Module,
    images: torch.Tensor,
    perturbation: Perturbation,
    property_fn: Callable[[torch.Tensor, torch.Tensor], List[bool]],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    """
    Golden section search for failure threshold.

    More efficient than binary search with ~38% reduction in evaluations.
    Uses the golden ratio to minimize function evaluations while
    maintaining logarithmic convergence.

    Args:
        model: Model to test
        images: Clean input images (N, C, H, W)
        perturbation: Perturbation to apply
        property_fn: Function(clean_logits, pert_logits) -> per-sample pass list
        config: Search configuration

    Returns:
        SearchResult with failure threshold and search history
    """
    config = config or SearchConfig()
    result = SearchResult()
    cache = EvaluationCache() if config.cache_evaluations else None

    # Get clean outputs once
    with torch.no_grad():
        model.eval()
        clean_outputs = model(images)

    a, b = config.min_level, config.max_level
    h = b - a

    # Initial interior points
    c = a + INVPHI2 * h
    d = a + INVPHI * h

    best_fail: Optional[float] = None
    best_pert_images: Optional[torch.Tensor] = None
    best_pert_outputs: Optional[torch.Tensor] = None

    # Evaluate initial points
    passed_c, frac_c, mask_c, pert_c, out_c, _ = _evaluate_with_cache(
        model, images, perturbation, c, property_fn, clean_outputs, cache, config.reduce
    )
    result.iterations += 1
    result.queries += 1

    passed_d, frac_d, mask_d, pert_d, out_d, _ = _evaluate_with_cache(
        model, images, perturbation, d, property_fn, clean_outputs, cache, config.reduce
    )
    result.iterations += 1
    result.queries += 1

    result.path.append({"level": c, "passed": passed_c, "pass_fraction": frac_c})
    result.path.append({"level": d, "passed": passed_d, "pass_fraction": frac_d})

    # Track failures
    if not passed_c:
        best_fail = c
        best_pert_images = pert_c
        best_pert_outputs = out_c
    if not passed_d and (best_fail is None or d < best_fail):
        best_fail = d
        best_pert_images = pert_d
        best_pert_outputs = out_d

    no_improvement_count = 0
    prev_best = best_fail

    while h > config.tolerance and result.iterations < config.max_iterations:
        if result.queries >= config.max_queries:
            break

        # Golden section logic: narrow the interval based on which point failed
        # We want to find the MINIMUM level that causes failure
        if passed_c and not passed_d:
            # Failure at d but not c: threshold is between c and d
            a = c
            c = d
            passed_c, frac_c, mask_c = passed_d, frac_d, mask_d
            h = b - a
            d = a + INVPHI * h

            passed_d, frac_d, mask_d, pert_d, out_d, cached = _evaluate_with_cache(
                model, images, perturbation, d, property_fn, clean_outputs, cache, config.reduce
            )
            if not cached:
                result.queries += 1
            result.iterations += 1
            result.path.append({"level": d, "passed": passed_d, "pass_fraction": frac_d})

            if not passed_d and (best_fail is None or d < best_fail):
                best_fail = d
                best_pert_images = pert_d
                best_pert_outputs = out_d

        elif not passed_c:
            # Failure at c: threshold is between a and c
            b = d
            d = c
            passed_d, frac_d, mask_d = passed_c, frac_c, mask_c
            h = b - a
            c = a + INVPHI2 * h

            passed_c, frac_c, mask_c, pert_c, out_c, cached = _evaluate_with_cache(
                model, images, perturbation, c, property_fn, clean_outputs, cache, config.reduce
            )
            if not cached:
                result.queries += 1
            result.iterations += 1
            result.path.append({"level": c, "passed": passed_c, "pass_fraction": frac_c})

            if not passed_c and (best_fail is None or c < best_fail):
                best_fail = c
                best_pert_images = pert_c
                best_pert_outputs = out_c

        else:
            # Both passed: increase search range
            a = d
            h = b - a
            c = a + INVPHI2 * h
            d = a + INVPHI * h

            passed_c, frac_c, mask_c, pert_c, out_c, cached = _evaluate_with_cache(
                model, images, perturbation, c, property_fn, clean_outputs, cache, config.reduce
            )
            if not cached:
                result.queries += 1
            result.iterations += 1
            result.path.append({"level": c, "passed": passed_c, "pass_fraction": frac_c})

            passed_d, frac_d, mask_d, pert_d, out_d, cached = _evaluate_with_cache(
                model, images, perturbation, d, property_fn, clean_outputs, cache, config.reduce
            )
            if not cached:
                result.queries += 1
            result.iterations += 1
            result.path.append({"level": d, "passed": passed_d, "pass_fraction": frac_d})

            if not passed_c and (best_fail is None or c < best_fail):
                best_fail = c
                best_pert_images = pert_c
                best_pert_outputs = out_c
            if not passed_d and (best_fail is None or d < best_fail):
                best_fail = d
                best_pert_images = pert_d
                best_pert_outputs = out_d

        # Early stopping check
        if best_fail == prev_best:
            no_improvement_count += 1
            if no_improvement_count >= config.early_stop_no_improvement:
                break
        else:
            no_improvement_count = 0
            prev_best = best_fail

    result.failure_threshold = best_fail
    result.converged = h <= config.tolerance
    result.perturbed_images = best_pert_images
    result.perturbed_output = best_pert_outputs
    result.cache_hits = cache.hits if cache else 0

    return result


def adaptive_search(
    model: nn.Module,
    images: torch.Tensor,
    perturbation: Perturbation,
    property_fn: Callable[[torch.Tensor, torch.Tensor], List[bool]],
    config: Optional[SearchConfig] = None,
    initial_step: float = 0.01,
) -> SearchResult:
    """
    Smart adaptive search with exponential exploration and refinement.

    Phase 1: Exponential exploration to find failure region
    Phase 2: Binary search refinement within the failure bracket

    This is more efficient than simple step-halving when the failure
    threshold is unknown and could be anywhere in the range.

    Args:
        model: Model to test
        images: Clean input images (N, C, H, W)
        perturbation: Perturbation to apply
        property_fn: Function(clean_logits, pert_logits) -> per-sample pass list
        config: Search configuration
        initial_step: Starting step size for exploration

    Returns:
        SearchResult with failure threshold and search history
    """
    config = config or SearchConfig()
    result = SearchResult()
    cache = EvaluationCache() if config.cache_evaluations else None

    # Get clean outputs once
    with torch.no_grad():
        model.eval()
        clean_outputs = model(images)

    # Per-sample tracking
    n_samples = images.shape[0]
    last_pass_levels: List[Optional[float]] = [None] * n_samples
    first_fail_levels: List[Optional[float]] = [None] * n_samples

    best_fail: Optional[float] = None
    best_pert_images: Optional[torch.Tensor] = None
    best_pert_outputs: Optional[torch.Tensor] = None

    # Phase 1: Exponential exploration to find bounds
    level = config.min_level + initial_step
    step = initial_step
    found_failure = False
    last_pass = config.min_level

    while level <= config.max_level and result.iterations < config.max_iterations // 2:
        if result.queries >= config.max_queries:
            break

        batch_passed, pass_frac, passed_mask, pert_imgs, pert_outs, cached = _evaluate_with_cache(
            model, images, perturbation, level, property_fn, clean_outputs, cache, config.reduce
        )

        if not cached:
            result.queries += 1
        result.iterations += 1

        result.path.append({
            "level": level,
            "passed": batch_passed,
            "pass_fraction": pass_frac,
            "phase": "exploration",
        })

        # Update per-sample brackets
        for i, passed in enumerate(passed_mask):
            if passed:
                last_pass_levels[i] = level
            elif first_fail_levels[i] is None:
                first_fail_levels[i] = level

        if batch_passed:
            last_pass = level
            # Exponential increase for faster exploration
            step *= 2
            level = level + step
        else:
            # Found failure - record and switch to refinement
            found_failure = True
            if best_fail is None or level < best_fail:
                best_fail = level
                best_pert_images = pert_imgs
                best_pert_outputs = pert_outs
            break

    # Phase 2: Binary search refinement if we found a failure bracket
    if found_failure and best_fail is not None:
        lo, hi = last_pass, best_fail

        while (hi - lo) > config.tolerance and result.iterations < config.max_iterations:
            if result.queries >= config.max_queries:
                break

            mid = (lo + hi) / 2

            batch_passed, pass_frac, passed_mask, pert_imgs, pert_outs, cached = _evaluate_with_cache(
                model, images, perturbation, mid, property_fn, clean_outputs, cache, config.reduce
            )

            if not cached:
                result.queries += 1
            result.iterations += 1

            result.path.append({
                "level": mid,
                "passed": batch_passed,
                "pass_fraction": pass_frac,
                "phase": "refinement",
            })

            # Update per-sample brackets
            for i, passed in enumerate(passed_mask):
                if passed:
                    last_pass_levels[i] = mid
                elif first_fail_levels[i] is None:
                    first_fail_levels[i] = mid

            if batch_passed:
                lo = mid
            else:
                hi = mid
                if mid < best_fail:
                    best_fail = mid
                    best_pert_images = pert_imgs
                    best_pert_outputs = pert_outs

    # If no failure found, try maximum level
    if best_fail is None and result.queries < config.max_queries:
        batch_passed, pass_frac, passed_mask, pert_imgs, pert_outs, cached = _evaluate_with_cache(
            model, images, perturbation, config.max_level, property_fn, clean_outputs, cache, config.reduce
        )

        if not cached:
            result.queries += 1
        result.iterations += 1

        result.path.append({
            "level": config.max_level,
            "passed": batch_passed,
            "pass_fraction": pass_frac,
            "phase": "final_check",
        })

        if not batch_passed:
            best_fail = config.max_level
            best_pert_images = pert_imgs
            best_pert_outputs = pert_outs

    # Finalize result
    result.failure_threshold = best_fail
    result.converged = best_fail is not None
    result.perturbed_images = best_pert_images
    result.perturbed_output = best_pert_outputs
    result.cache_hits = cache.hits if cache else 0

    # Per-sample threshold estimates (midpoint of bracket)
    result.per_sample_thresholds = [
        0.5 * (lo + hi) if lo is not None and hi is not None else None
        for lo, hi in zip(last_pass_levels, first_fail_levels)
    ]

    return result


def binary_search(
    model: nn.Module,
    images: torch.Tensor,
    perturbation: Perturbation,
    property_fn: Callable[[torch.Tensor, torch.Tensor], List[bool]],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    """
    Binary search for failure threshold.

    Efficiently finds the minimum perturbation level causing failure
    using O(log n) binary search.

    Args:
        model: Model to test
        images: Clean input images (N, C, H, W)
        perturbation: Perturbation to apply
        property_fn: Function(clean_logits, pert_logits) -> per-sample pass list
        config: Search configuration

    Returns:
        SearchResult with failure threshold and search history
    """
    config = config or SearchConfig()
    result = SearchResult()
    cache = EvaluationCache() if config.cache_evaluations else None

    lo, hi = config.min_level, config.max_level
    best_fail: Optional[float] = None
    best_pert_images: Optional[torch.Tensor] = None
    best_pert_outputs: Optional[torch.Tensor] = None

    # Get clean outputs once
    with torch.no_grad():
        model.eval()
        clean_outputs = model(images)

    # Per-sample tracking
    n_samples = images.shape[0]
    last_pass_levels: List[Optional[float]] = [None] * n_samples
    first_fail_levels: List[Optional[float]] = [None] * n_samples

    while (hi - lo) > config.tolerance and result.iterations < config.max_iterations:
        if result.queries >= config.max_queries:
            break

        level = (lo + hi) / 2.0

        batch_passed, pass_frac, passed_mask, pert_imgs, pert_outs, cached = _evaluate_with_cache(
            model, images, perturbation, level, property_fn, clean_outputs, cache, config.reduce
        )

        if not cached:
            result.queries += 1
        result.iterations += 1

        # Update per-sample brackets
        for i, passed in enumerate(passed_mask):
            if passed:
                last_pass_levels[i] = level
            elif first_fail_levels[i] is None:
                first_fail_levels[i] = level

        result.path.append({
            "level": level,
            "passed": batch_passed,
            "pass_fraction": pass_frac,
            "passed_mask": passed_mask,
        })

        if batch_passed:
            lo = level
        else:
            hi = level
            if best_fail is None or level < best_fail:
                best_fail = level
                best_pert_images = pert_imgs
                best_pert_outputs = pert_outs

    result.failure_threshold = best_fail
    result.converged = (hi - lo) <= config.tolerance
    result.perturbed_images = best_pert_images
    result.perturbed_output = best_pert_outputs
    result.cache_hits = cache.hits if cache else 0

    result.per_sample_thresholds = [
        0.5 * (lo + hi) if lo is not None and hi is not None else None
        for lo, hi in zip(last_pass_levels, first_fail_levels)
    ]

    return result


def grid_search(
    model: nn.Module,
    images: torch.Tensor,
    perturbation: Perturbation,
    property_fn: Callable[[torch.Tensor, torch.Tensor], List[bool]],
    config: Optional[SearchConfig] = None,
    num_levels: int = 21,
) -> SearchResult:
    """
    Grid search for failure threshold.

    Tests evenly-spaced levels and returns the first failure point.

    Args:
        model: Model to test
        images: Clean input images (N, C, H, W)
        perturbation: Perturbation to apply
        property_fn: Function(clean_logits, pert_logits) -> per-sample pass list
        config: Search configuration
        num_levels: Number of levels to test

    Returns:
        SearchResult with failure threshold and search history
    """
    config = config or SearchConfig()
    result = SearchResult()

    best_fail: Optional[float] = None
    best_pert_images: Optional[torch.Tensor] = None
    best_pert_outputs: Optional[torch.Tensor] = None

    # Get clean outputs once
    with torch.no_grad():
        model.eval()
        clean_outputs = model(images)

    # Generate test levels
    levels = torch.linspace(config.min_level, config.max_level, num_levels)

    for level in levels:
        if result.queries >= config.max_queries:
            break

        level = float(level)

        pert_imgs, pert_outs, passed_mask = _evaluate_batch(
            model, images, perturbation, level, property_fn, clean_outputs
        )
        batch_passed, pass_frac = _reduce_passed(passed_mask, config.reduce)

        result.path.append({
            "level": level,
            "passed": batch_passed,
            "pass_fraction": pass_frac,
        })

        result.iterations += 1
        result.queries += 1

        if not batch_passed and best_fail is None:
            best_fail = level
            best_pert_images = pert_imgs
            best_pert_outputs = pert_outs

    result.failure_threshold = best_fail
    result.converged = True
    result.perturbed_images = best_pert_images
    result.perturbed_output = best_pert_outputs

    return result


# ============================================================
# Property Functions for common robustness checks
# ============================================================

def label_preserved(
    clean_logits: torch.Tensor,
    pert_logits: torch.Tensor,
) -> List[bool]:
    """
    Check if predicted label is preserved.

    Args:
        clean_logits: Clean model outputs (N, num_classes)
        pert_logits: Perturbed model outputs (N, num_classes)

    Returns:
        Per-sample list of whether label was preserved
    """
    clean_preds = torch.argmax(clean_logits, dim=-1)
    pert_preds = torch.argmax(pert_logits, dim=-1)
    return (clean_preds == pert_preds).tolist()


def confidence_above(threshold: float = 0.5) -> Callable[[torch.Tensor, torch.Tensor], List[bool]]:
    """
    Create property function that checks if confidence stays above threshold.

    Args:
        threshold: Minimum confidence required

    Returns:
        Property function for use with search algorithms
    """
    def check(_clean_logits: torch.Tensor, pert_logits: torch.Tensor) -> List[bool]:
        pert_probs = torch.softmax(pert_logits, dim=-1)
        max_conf = pert_probs.max(dim=-1).values
        return (max_conf >= threshold).tolist()

    return check


def top_k_preserved(k: int = 5) -> Callable[[torch.Tensor, torch.Tensor], List[bool]]:
    """
    Create property function that checks if top-k predictions are preserved.

    Args:
        k: Number of top predictions to check

    Returns:
        Property function for use with search algorithms
    """
    def check(clean_logits: torch.Tensor, pert_logits: torch.Tensor) -> List[bool]:
        _, clean_topk = torch.topk(clean_logits, k, dim=-1)
        _, pert_topk = torch.topk(pert_logits, k, dim=-1)

        results = []
        for c, p in zip(clean_topk, pert_topk):
            preserved = c[0].item() in p.tolist()
            results.append(preserved)

        return results

    return check


# ============================================================
# Integration helpers for visprobe runner
# ============================================================

def create_property_fn_from_user_func(
    user_func: Callable,
    batch_size: int,
) -> Callable[[torch.Tensor, torch.Tensor], List[bool]]:
    """
    Create a property function from a visprobe user-defined test function.

    Args:
        user_func: User's decorated test function
        batch_size: Batch size for evaluation

    Returns:
        Property function compatible with search algorithms
    """
    def property_fn(clean_logits: torch.Tensor, pert_logits: torch.Tensor) -> List[bool]:
        # Handle different function signatures
        try:
            # Try vectorized call first
            result = user_func(clean_logits, pert_logits)
            if isinstance(result, torch.Tensor):
                return result.bool().tolist()
            elif isinstance(result, (list, tuple)):
                return [bool(r) for r in result]
            else:
                return [bool(result)] * batch_size
        except Exception:
            # Fallback to per-sample evaluation
            results = []
            for i in range(batch_size):
                try:
                    r = user_func(clean_logits[i:i+1], pert_logits[i:i+1])
                    results.append(bool(r))
                except Exception:
                    results.append(False)
            return results

    return property_fn
