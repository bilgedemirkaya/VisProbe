"""
Core search engine for finding failure thresholds.

This module contains the unified search algorithm used by both quick_check()
and search() functions.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from ..strategies.base import Strategy


class SearchEngine:
    """
    Finds failure threshold for a given strategy using adaptive search.

    Used by:
    - quick_check(): Runs SearchEngine for each strategy in a preset
    - search(): Runs SearchEngine for a single user-specified strategy

    Args:
        model: PyTorch model to test
        strategy_factory: Factory function that creates a strategy at a given level
        property_fn: Function that evaluates if property holds (returns True if passes)
        samples: List of (image, label) tuples to test
        mode: Search mode - "adaptive" (step-halving) or "binary"
        level_lo: Lower bound for perturbation level
        level_hi: Upper bound for perturbation level
        initial_level: Starting level for adaptive search
        step: Initial step size for adaptive search
        min_step: Minimum step size before stopping
        max_queries: Maximum number of search iterations
        pass_threshold: Fraction of samples that must pass (default 0.9 = 90%)
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        strategy_factory: Callable[[float], Strategy],
        property_fn: Callable,
        samples: List[Tuple[torch.Tensor, int]],
        mode: str = "adaptive",
        level_lo: float = 0.0,
        level_hi: float = 1.0,
        initial_level: Optional[float] = None,
        step: Optional[float] = None,
        min_step: float = 0.001,
        max_queries: int = 100,
        pass_threshold: float = 0.9,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.strategy_factory = strategy_factory
        self.property_fn = property_fn
        self.samples = samples
        self.mode = mode
        self.level_lo = level_lo
        self.level_hi = level_hi
        self.initial_level = initial_level if initial_level is not None else level_lo
        self.step = step if step is not None else (level_hi - level_lo) / 10.0
        self.min_step = min_step
        self.max_queries = max_queries
        self.pass_threshold = pass_threshold
        self.device = device or torch.device("cpu")

        # Validate inputs
        if mode not in {"adaptive", "binary"}:
            raise ValueError(f"Unknown mode: {mode}. Must be 'adaptive' or 'binary'")
        if level_lo >= level_hi:
            raise ValueError(f"level_lo ({level_lo}) must be < level_hi ({level_hi})")
        if len(samples) == 0:
            raise ValueError("samples list cannot be empty")

    def run(self, progress_bar: Optional[tqdm] = None) -> Dict[str, Any]:
        """
        Run search to find failure threshold.

        Args:
            progress_bar: Optional tqdm progress bar for updates

        Returns:
            dict with:
            - failure_threshold: float (level where model starts failing)
            - last_pass_level: float (highest level where model passed)
            - robustness_score: float (0-1, normalized score)
            - queries: int (number of search iterations used)
            - failures: list (failure cases with details)
            - search_path: list (history of search iterations)
            - runtime: float (seconds)
        """
        start_time = time.time()

        if self.mode == "adaptive":
            result = self._adaptive_search(progress_bar)
        elif self.mode == "binary":
            result = self._binary_search(progress_bar)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        result["runtime"] = time.time() - start_time
        return result

    def _adaptive_search(self, progress_bar: Optional[tqdm] = None) -> Dict[str, Any]:
        """
        Adaptive step-halving search.

        Algorithm:
        1. Start at initial_level
        2. Test if property holds at this level
        3. If passes: increase level by step
        4. If fails: record failure, decrease level by step/2, halve step
        5. Stop when step < min_step or queries >= max_queries
        """
        self.model.eval()

        current_level = self.initial_level
        step_size = self.step
        queries = 0
        failures: List[Dict[str, Any]] = []
        search_path: List[Dict[str, Any]] = []

        last_pass_level = self.level_lo
        first_fail_level = self.level_hi

        while queries < self.max_queries and step_size >= self.min_step:
            queries += 1

            # Evaluate at current level
            passed, pass_rate, level_failures = self._evaluate_at_level(current_level)

            search_path.append({
                "iteration": queries,
                "level": current_level,
                "passed": passed,
                "pass_rate": pass_rate,
            })

            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "level": f"{current_level:.3f}",
                    "pass_rate": f"{pass_rate:.2%}"
                })

            if passed:
                # Property holds - increase level
                last_pass_level = current_level
                current_level += step_size
                if current_level > self.level_hi:
                    current_level = self.level_hi
                    break
            else:
                # Property fails - record and decrease
                first_fail_level = current_level
                failures.extend(level_failures)
                current_level -= step_size / 2.0
                step_size /= 2.0

                if current_level < self.level_lo:
                    current_level = self.level_lo
                    break

        # Calculate robustness score (0-1, higher is better)
        if self.level_hi > self.level_lo:
            score = (last_pass_level - self.level_lo) / (self.level_hi - self.level_lo)
        else:
            score = 1.0

        return {
            "failure_threshold": first_fail_level,
            "last_pass_level": last_pass_level,
            "robustness_score": min(max(score, 0.0), 1.0),
            "queries": queries,
            "failures": failures[:10],  # Keep top 10
            "search_path": search_path,
        }

    def _binary_search(self, progress_bar: Optional[tqdm] = None) -> Dict[str, Any]:
        """
        Binary search over [level_lo, level_hi].

        Algorithm:
        1. Test midpoint = (lo + hi) / 2
        2. If fails: search in [lo, midpoint]
        3. If passes: search in [midpoint, hi]
        4. Stop when hi - lo < tolerance or queries >= max_queries
        """
        self.model.eval()

        lo = self.level_lo
        hi = self.level_hi
        queries = 0
        failures: List[Dict[str, Any]] = []
        search_path: List[Dict[str, Any]] = []

        last_pass_level = lo
        first_fail_level = hi
        tolerance = self.min_step

        while queries < self.max_queries and (hi - lo) >= tolerance:
            queries += 1
            midpoint = (lo + hi) / 2.0

            # Evaluate at midpoint
            passed, pass_rate, level_failures = self._evaluate_at_level(midpoint)

            search_path.append({
                "iteration": queries,
                "level": midpoint,
                "passed": passed,
                "pass_rate": pass_rate,
                "lo": lo,
                "hi": hi,
            })

            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "level": f"{midpoint:.4f}",
                    "pass_rate": f"{pass_rate:.2%}",
                    "range": f"[{lo:.4f}, {hi:.4f}]"
                })

            if passed:
                # Property holds at midpoint - search higher
                last_pass_level = midpoint
                lo = midpoint
            else:
                # Property fails at midpoint - search lower
                first_fail_level = midpoint
                failures.extend(level_failures)
                hi = midpoint

        # Calculate robustness score
        if self.level_hi > self.level_lo:
            score = (last_pass_level - self.level_lo) / (self.level_hi - self.level_lo)
        else:
            score = 1.0

        return {
            "failure_threshold": first_fail_level,
            "last_pass_level": last_pass_level,
            "robustness_score": min(max(score, 0.0), 1.0),
            "queries": queries,
            "failures": failures[:10],
            "search_path": search_path,
        }

    def _evaluate_at_level(
        self, level: float
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Test if property holds at given perturbation level.

        Args:
            level: Perturbation level to test

        Returns:
            (passed: bool, pass_rate: float, failures: list)
            - passed: True if pass_rate >= pass_threshold
            - pass_rate: Fraction of samples that passed
            - failures: List of failure cases
        """
        # Instantiate strategy at this level
        strategy = self.strategy_factory(level)

        passed_count = 0
        failed_samples: List[Dict[str, Any]] = []

        for idx, (img, label) in enumerate(self.samples):
            # Get original prediction
            with torch.no_grad():
                orig_out = self.model(img.unsqueeze(0))
                orig_pred = torch.argmax(orig_out, dim=-1).item()

            # Apply perturbation
            perturbed = strategy.generate(img.unsqueeze(0), self.model, level=level)

            # Get perturbed prediction
            with torch.no_grad():
                pert_out = self.model(perturbed)
                pert_pred = torch.argmax(pert_out, dim=-1).item()

            # Check property (default: label constant)
            if orig_pred == pert_pred:
                passed_count += 1
            else:
                failed_samples.append({
                    "index": idx,
                    "level": level,
                    "original_pred": orig_pred,
                    "perturbed_pred": pert_pred,
                    "original_label": label,
                })

        pass_rate = passed_count / len(self.samples)
        passed = pass_rate >= self.pass_threshold

        return passed, pass_rate, failed_samples
