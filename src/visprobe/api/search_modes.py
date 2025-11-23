"""
Search mode implementations (adaptive, grid, random, binary) extracted from runner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

from ..strategies.base import Strategy  # noqa: E402
from .model_wrap import _ModelWithIntermediateOutput  # noqa: E402
from .query_counter import QueryCounter  # noqa: E402


def resolve_strategy_for_level(runner, params: Dict[str, Any], level: float):
    strategy_spec = params["strategy"]
    perturb_obj = None
    if callable(strategy_spec):
        try:
            candidate = strategy_spec(level)
            if callable(getattr(candidate, "generate", None)):
                perturb_obj = candidate
        except TypeError:
            perturb_obj = None
        if perturb_obj is None:
            perturb_obj = Strategy.resolve(strategy_spec, level=level)
    else:
        perturb_obj = Strategy.resolve(strategy_spec)
        if hasattr(perturb_obj, "eps"):
            perturb_obj.eps = level
        elif hasattr(perturb_obj, "level"):
            perturb_obj.level = level
    return perturb_obj


def _compute_batch_pass_from_mask(passed_mask: List[bool], reducer_spec: Any) -> Tuple[bool, float]:
    """Returns (batch_pass, passed_fraction) using the configured reducer."""
    if not passed_mask:
        return False, 0.0
    try:
        import torch as _torch

        passed_frac = _torch.tensor(passed_mask, dtype=_torch.float32).mean().item()
    except Exception:
        passed_frac = sum(1 for v in passed_mask if v) / max(1, len(passed_mask))

    reducer = reducer_spec or "all"
    if reducer == "all":
        return all(passed_mask), passed_frac
    if reducer == "any":
        return any(passed_mask), passed_frac
    if isinstance(reducer, str) and reducer.startswith("frac>="):
        try:
            thr = float(reducer.split(">=", 1)[1])
        except Exception:
            thr = 1.0
        return passed_frac >= thr, passed_frac
    # default
    return all(passed_mask), passed_frac


def perform_adaptive_search(  # noqa: C901
    runner, params: Dict[str, Any], clean_results: Tuple
) -> Dict:
    """Step-halving adaptive search with per-sample bracketing and full path logging."""
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, clean_feat = clean_results
    top_k = params.get("top_k")

    level = float(params.get("initial_level", 0.001))
    step = float(params.get("step", 0.002))
    min_step = float(params.get("min_step", 1e-5))
    max_queries = int(params.get("max_queries", 500))
    level_lo = params.get("level_lo", None)
    level_hi = params.get("level_hi", None)
    reducer_spec = params.get("reduce", "all")

    path: List[Dict[str, Any]] = []
    top_k_path: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"failure_threshold": None}
    best_fail: Optional[float] = None

    # Per-image bracketing
    try:
        batch_size_local = int(clean_logits.shape[0])
    except Exception:
        batch_size_local = int(ctx["batch_size"])
    last_pass_levels: List[Optional[float]] = [None] * batch_size_local
    first_fail_levels: List[Optional[float]] = [None] * batch_size_local

    # Prepare model state once
    m.eval()
    previous_level: Optional[float] = None
    first_failing_index: Optional[int] = None

    while step > min_step and runner.query_count < max_queries:
        # Clamp level to domain if provided
        if level_lo is not None:
            level = max(float(level), float(level_lo))
        else:
            level = max(float(level), 0.0)
        if level_hi is not None:
            level = min(float(level), float(level_hi))

        # Resolve and configure strategy for this level
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        try:
            perturb_obj.configure(
                mean=runner.context["mean"],
                std=runner.context["std"],
                seed=runner.context.get("seed"),
            )
        except Exception:
            try:
                from .runner import TestRunner as _TR

                _TR._configure_strategy(
                    perturb_obj, runner.context["mean"], runner.context["std"], None
                )
            except Exception:
                pass

        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, pert_feat_i = pert_model_output
            else:
                pert_out_i, pert_feat_i = pert_model_output, None

            # Evaluate property mask once; use logits-only
            passed_mask: List[bool] = runner._evaluate_property_mask(
                clean_logits, runner._to_logits(pert_out_i)
            )
            batch_pass, passed_frac = _compute_batch_pass_from_mask(passed_mask, reducer_spec)

        runner.query_count += qc.extra

        # Log predictions/confidence (first sample for summary, full lists for details)
        pert_pred_labels = torch.argmax(pert_out_i, dim=-1)
        pred_indices = pert_pred_labels.tolist()
        conf_list = torch.softmax(pert_out_i, dim=-1).max(dim=-1).values.tolist()
        path.append(
            {
                "level": float(level),
                "passed": batch_pass,
                "passed_all": all(passed_mask),
                "passed_frac": float(passed_frac),
                "passed_mask": passed_mask,
                "prediction": (
                    ctx["class_names"][pred_indices[0]]
                    if ctx["class_names"]
                    else str(pred_indices[0])
                ),
                "confidence": conf_list[0],
                "predictions": [
                    ctx["class_names"][i] if ctx["class_names"] else str(i) for i in pred_indices
                ],
                "confidences": conf_list,
            }
        )

        if top_k:
            from .analysis_utils import run_top_k_analysis as _run_tk

            overlap = _run_tk(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": float(level), "overlap": overlap})

        # Update per-sample brackets using the property mask
        for si, ok in enumerate(passed_mask):
            if ok:
                last_pass_levels[si] = float(level)
            elif first_fail_levels[si] is None:
                first_fail_levels[si] = float(level)
                if first_failing_index is None:
                    first_failing_index = si

        # Maintain minimal failing level details
        if not batch_pass:
            if best_fail is None or level < best_fail:
                best_fail = float(level)
                result.update(
                    {
                        "failure_threshold": best_fail,
                        "perturbed_tensor": pert_tensor,
                        "perturbed_output": pert_out_i,
                        "perturbed_features": pert_feat_i,
                    }
                )
            # Step-halving on failure
            level_next = level - step
            step *= 0.5
        else:
            level_next = level + step

        # Guard against non-progress due to numerical issues
        if previous_level is not None and abs(level_next - previous_level) < 1e-12:
            break
        previous_level = level_next
        level = level_next

        # Optional early exit if we already found effectively zero threshold
        if best_fail is not None and best_fail <= 0.0:
            break

    # Finalize output
    result["failure_threshold"] = best_fail
    result["path"] = path
    result["top_k_path"] = top_k_path
    try:
        result["first_failing_index"] = int(
            first_failing_index if first_failing_index is not None else 0
        )
    except Exception:
        result["first_failing_index"] = 0

    # Midpoint estimate per image when bracketed
    per_sample_thresholds: List[Optional[float]] = [
        (0.5 * (float(lo) + float(hi))) if (lo is not None and hi is not None) else None
        for lo, hi in zip(last_pass_levels, first_fail_levels)
    ]
    result["per_sample_thresholds"] = per_sample_thresholds
    return result


def perform_grid_search(runner, params: Dict[str, Any], clean_results: Tuple) -> Dict:  # noqa: C901
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, _ = clean_results
    top_k = params.get("top_k")

    level_lo = float(params.get("level_lo", 0.0))
    level_hi = float(params.get("level_hi", 0.2))
    num_levels = int(params.get("num_levels", 21))
    max_queries = int(params.get("max_queries", 500))

    levels = np.linspace(level_lo, level_hi, num_levels)
    path, top_k_path = [], []
    result: Dict[str, Any] = {"failure_threshold": None}

    for level in levels:
        if runner.query_count >= max_queries:
            break
        level = float(max(level, 0.0))
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        try:
            perturb_obj.configure(
                mean=runner.context["mean"],
                std=runner.context["std"],
                seed=runner.context.get("seed"),
            )
        except Exception:
            try:
                from .runner import TestRunner as _TR

                _TR._configure_strategy(
                    perturb_obj, runner.context["mean"], runner.context["std"], None
                )
            except Exception:
                pass

        m.eval()
        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, _ = pert_model_output
            else:
                pert_out_i = pert_model_output

            passed = (
                runner._evaluate_property(
                    clean_logits, runner._to_logits(pert_out_i), vectorized=True
                )
                > 0
            )

        runner.query_count += qc.extra

        pred_indices = torch.argmax(pert_out_i, dim=-1).tolist()
        conf_list = torch.softmax(pert_out_i, dim=-1).max(dim=-1).values.tolist()
        path.append(
            {
                "level": float(level),
                "passed": passed,
                "prediction": (
                    ctx["class_names"][pred_indices[0]]
                    if ctx["class_names"]
                    else str(pred_indices[0])
                ),
                "confidence": conf_list[0],
                "predictions": [
                    ctx["class_names"][i] if ctx["class_names"] else str(i) for i in pred_indices
                ],
                "confidences": conf_list,
            }
        )

        from .analysis_utils import run_top_k_analysis

        if top_k:
            overlap = run_top_k_analysis(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": level, "overlap": overlap})

        if not passed and result["failure_threshold"] is None:
            result.update(
                {
                    "failure_threshold": float(level),
                    "perturbed_tensor": pert_tensor,
                    "perturbed_output": pert_out_i,
                }
            )
            break

    result["path"] = path
    result["top_k_path"] = top_k_path
    return result


def perform_random_search(  # noqa: C901
    runner, params: Dict[str, Any], clean_results: Tuple
) -> Dict:
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, _ = clean_results
    top_k = params.get("top_k")

    level_lo = float(params.get("level_lo", 0.0))
    level_hi = float(params.get("level_hi", 0.2))
    num_samples = int(params.get("num_samples", 64))
    max_queries = int(params.get("max_queries", 500))

    rng = np.random.default_rng(params.get("random_seed", None))
    levels = rng.uniform(level_lo, level_hi, size=num_samples)
    path, top_k_path = [], []
    fail_levels: List[float] = []
    best_fail = None
    result: Dict[str, Any] = {"failure_threshold": None}

    for level in levels:
        if runner.query_count >= max_queries:
            break
        level = float(max(level, 0.0))
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        try:
            perturb_obj.configure(
                mean=runner.context["mean"],
                std=runner.context["std"],
                seed=runner.context.get("seed"),
            )
        except Exception:
            try:
                from .runner import TestRunner as _TR

                _TR._configure_strategy(
                    perturb_obj, runner.context["mean"], runner.context["std"], None
                )
            except Exception:
                pass

        m.eval()
        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, _ = pert_model_output
            else:
                pert_out_i = pert_model_output

            passed = (
                runner._evaluate_property(
                    clean_logits, runner._to_logits(pert_out_i), vectorized=True
                )
                > 0
            )

        runner.query_count += qc.extra

        pred_indices = torch.argmax(pert_out_i, dim=-1).tolist()
        conf_list = torch.softmax(pert_out_i, dim=-1).max(dim=-1).values.tolist()
        path.append(
            {
                "level": float(level),
                "passed": passed,
                "prediction": (
                    ctx["class_names"][pred_indices[0]]
                    if ctx["class_names"]
                    else str(pred_indices[0])
                ),
                "confidence": conf_list[0],
                "predictions": [
                    ctx["class_names"][i] if ctx["class_names"] else str(i) for i in pred_indices
                ],
                "confidences": conf_list,
            }
        )

        from .analysis_utils import run_top_k_analysis

        if top_k:
            overlap = run_top_k_analysis(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": level, "overlap": overlap})

        if not passed:
            fail_levels.append(float(level))
            if best_fail is None or level < best_fail:
                best_fail = float(level)
                result.update(
                    {
                        "failure_threshold": best_fail,
                        "perturbed_tensor": pert_tensor,
                        "perturbed_output": pert_out_i,
                    }
                )

    result["path"] = path
    result["top_k_path"] = top_k_path
    result["fail_levels"] = fail_levels
    return result


def perform_binary_search(  # noqa: C901
    runner, params: Dict[str, Any], clean_results: Tuple
) -> Dict:
    """
    Optimized binary search for finding failure threshold.

    More efficient than adaptive search for finding the exact failure point.
    Uses true binary search with O(log n) complexity.
    """
    ctx = runner.context
    m, batch_tensor = ctx["model"], ctx["batch_tensor"]
    clean_logits, clean_feat = clean_results
    top_k = params.get("top_k")

    # Binary search bounds
    lo = float(params.get("level_lo", 0.0))
    hi = float(params.get("level_hi", 1.0))
    min_step = float(params.get("min_step", 1e-5))
    max_queries = int(params.get("max_queries", 500))
    reducer_spec = params.get("reduce", "all")

    path: List[Dict[str, Any]] = []
    top_k_path: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"failure_threshold": None}
    best_fail: Optional[float] = None

    # Per-image bracketing
    try:
        batch_size_local = int(clean_logits.shape[0])
    except Exception:
        batch_size_local = int(ctx["batch_size"])
    last_pass_levels: List[Optional[float]] = [None] * batch_size_local
    first_fail_levels: List[Optional[float]] = [None] * batch_size_local

    m.eval()
    first_failing_index: Optional[int] = None

    logger.debug(f"Starting binary search between {lo} and {hi}")

    while (hi - lo) > min_step and runner.query_count < max_queries:
        # Binary search midpoint
        level = (lo + hi) / 2.0

        # Resolve and configure strategy for this level
        perturb_obj = resolve_strategy_for_level(runner, params, level)
        try:
            perturb_obj.configure(
                mean=runner.context["mean"],
                std=runner.context["std"],
                seed=runner.context.get("seed"),
            )
        except Exception:
            try:
                from .runner import TestRunner as _TR

                _TR._configure_strategy(
                    perturb_obj, runner.context["mean"], runner.context["std"], None
                )
            except Exception:
                pass

        with QueryCounter(m) as qc:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m, level=level)
            pert_model_output = m(pert_tensor)
            if isinstance(m, _ModelWithIntermediateOutput):
                pert_out_i, pert_feat_i = pert_model_output
            else:
                pert_out_i, pert_feat_i = pert_model_output, None

            # Evaluate property mask
            passed_mask: List[bool] = runner._evaluate_property_mask(
                clean_logits, runner._to_logits(pert_out_i)
            )
            batch_pass, passed_frac = _compute_batch_pass_from_mask(passed_mask, reducer_spec)

        runner.query_count += qc.extra

        # Log this iteration
        pert_pred_labels = torch.argmax(pert_out_i, dim=-1)
        pred_indices = pert_pred_labels.tolist()
        conf_list = torch.softmax(pert_out_i, dim=-1).max(dim=-1).values.tolist()
        path.append(
            {
                "level": float(level),
                "passed": batch_pass,
                "passed_all": all(passed_mask),
                "passed_frac": float(passed_frac),
                "passed_mask": passed_mask,
                "prediction": (
                    ctx["class_names"][pred_indices[0]]
                    if ctx["class_names"]
                    else str(pred_indices[0])
                ),
                "confidence": conf_list[0],
                "predictions": [
                    ctx["class_names"][i] if ctx["class_names"] else str(i) for i in pred_indices
                ],
                "confidences": conf_list,
            }
        )

        if top_k:
            from .analysis_utils import run_top_k_analysis as _run_tk

            overlap = _run_tk(runner, top_k, clean_logits, runner._to_logits(pert_out_i))
            if overlap is not None:
                top_k_path.append({"level": float(level), "overlap": overlap})

        # Update per-sample brackets
        for si, ok in enumerate(passed_mask):
            if ok:
                last_pass_levels[si] = float(level)
            elif first_fail_levels[si] is None:
                first_fail_levels[si] = float(level)
                if first_failing_index is None:
                    first_failing_index = si

        # Binary search logic
        if batch_pass:
            # All passed, increase perturbation
            lo = level
            logger.debug(f"Level {level:.6f} passed, searching higher")
        else:
            # Failed, decrease perturbation
            hi = level
            if best_fail is None or level < best_fail:
                best_fail = float(level)
                result.update(
                    {
                        "failure_threshold": best_fail,
                        "perturbed_tensor": pert_tensor,
                        "perturbed_output": pert_out_i,
                        "perturbed_features": pert_feat_i,
                    }
                )
            logger.debug(f"Level {level:.6f} failed, searching lower")

    logger.info(
        f"Binary search completed in {len(path)} iterations, failure threshold: {best_fail}"
    )

    # Finalize output
    result["failure_threshold"] = best_fail
    result["path"] = path
    result["top_k_path"] = top_k_path
    try:
        result["first_failing_index"] = int(
            first_failing_index if first_failing_index is not None else 0
        )
    except Exception:
        result["first_failing_index"] = 0

    # Midpoint estimate per image when bracketed
    per_sample_thresholds: List[Optional[float]] = [
        (0.5 * (float(lo) + float(hi))) if (lo is not None and hi is not None) else None
        for lo, hi in zip(last_pass_levels, first_fail_levels)
    ]
    result["per_sample_thresholds"] = per_sample_thresholds
    return result
