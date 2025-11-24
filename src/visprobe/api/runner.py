"""
This module contains the core TestRunner class that orchestrates test execution,
including running perturbations, performing analysis, and generating reports.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from torchvision.transforms.functional import resize
except ImportError:
    print(
        "Warning: torchvision is not installed. "
        "Some features like resolution analysis may not be available."
    )

    def resize(img, size):
        raise NotImplementedError("torchvision is required for resolution analysis.")


import subprocess  # noqa: E402

import torch as _torch_version_probe  # noqa: E402

from ..strategies.base import Strategy  # noqa: E402
from .config import (  # noqa: E402
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    configure_threading,
    get_default_device,
)
from .report import ImageData, PanelImage, PerturbationInfo, Report  # noqa: E402

try:
    import torchvision as _torchvision_version_probe  # noqa: E402
except Exception:
    _torchvision_version_probe = None

from .analysis_utils import (  # noqa: E402
    run_corruption_sweep,
    run_ensemble_analysis,
    run_noise_sweep,
    run_resolution_impact,
    run_top_k_analysis,
)
from .model_wrap import _ModelWithIntermediateOutput  # noqa: E402
from .query_counter import QueryCounter  # noqa: E402
from .search_modes import (  # noqa: E402
    perform_adaptive_search,
    perform_binary_search,
    perform_grid_search,
    perform_random_search,
)
from .utils import build_final_strategy, build_search_blocks, build_visuals  # noqa: E402


class TestRunner:
    """Orchestrates a VisProbe test run, from setup to reporting."""

    def __init__(self, user_func: Callable, test_type: str, params: Dict[str, Any]):
        self.user_func = user_func
        self.test_type = test_type
        self.params = params
        self.query_count = 0
        self.start_time = time.perf_counter()
        self.context = self._get_test_context()

    def _get_test_context(self) -> Dict[str, Any]:  # noqa: C901
        """
        Gathers all necessary context from the decorated user function and
        places model/data on device.
        """
        # Configure threading for stability (only if safe)
        try:
            configure_threading()
        except Exception as e:
            # Skip threading config if it causes issues
            logger.warning(f"Failed to configure threading: {e}")

        model_obj = getattr(self.user_func, "_visprobe_model")
        capture_layers = getattr(self.user_func, "_visprobe_capture_intermediate_layers", None)
        wrapped_model = (
            _ModelWithIntermediateOutput(model_obj, capture_layers) if capture_layers else model_obj
        )
        data_obj = getattr(self.user_func, "_visprobe_data")
        collate_fn = getattr(self.user_func, "_visprobe_collate", None)
        batch_tensor = collate_fn(data_obj) if callable(collate_fn) else data_obj

        # --- Smart device selection with stability prioritization ---
        device = get_default_device()

        # Move model to device with better error handling
        try:
            if hasattr(wrapped_model, "to"):
                wrapped_model = wrapped_model.to(device)
            elif hasattr(wrapped_model, "model") and hasattr(wrapped_model.model, "to"):
                # Handle wrapped models
                wrapped_model.model = wrapped_model.model.to(device)
        except RuntimeError as e:
            logger.warning(f"Could not move model to {device}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error moving model to {device}: {e}")

        # Move data to device
        try:
            if hasattr(batch_tensor, "to"):
                batch_tensor = batch_tensor.to(device)
        except RuntimeError as e:
            logger.warning(f"Could not move data to {device}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error moving data to {device}: {e}")

        # RNG/seed for deterministic strategy behavior when supported
        try:
            seed = int(os.environ.get("VISPROBE_SEED", os.environ.get("RQ2_SEED", "1337")))
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid seed value in environment, using default 1337: {e}")
            seed = 1337

        rng = torch.Generator()
        try:
            rng.manual_seed(seed)
        except Exception as e:
            logger.warning(f"Could not set random seed {seed}: {e}")

        # robust batch size detection
        try:
            batch_size = int(getattr(batch_tensor, "shape", [1])[0])
        except (AttributeError, IndexError, ValueError) as e:
            logger.warning(f"Could not detect batch size, defaulting to 1: {e}")
            batch_size = 1

        return {
            "model": wrapped_model,
            "batch_tensor": batch_tensor,
            "batch_size": batch_size,
            "class_names": getattr(self.user_func, "_visprobe_class_names", None),
            "mean": getattr(self.user_func, "_visprobe_mean", IMAGENET_DEFAULT_MEAN),
            "std": getattr(self.user_func, "_visprobe_std", IMAGENET_DEFAULT_STD),
            "capture_layers": capture_layers,
            "device": str(device),
            "rng": rng,
            "seed": seed,
        }

    def run(self) -> Report:
        """Executes the test and returns a report."""
        if self.test_type == "given":
            report = self._run_given()
        elif self.test_type == "search":
            report = self._run_search()
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
        report.save()
        return report

    def _run_given(self) -> Report:
        ctx, p = self.context, self.params
        m, batch_tensor, batch_size = ctx["model"], ctx["batch_tensor"], ctx["batch_size"]

        # Clean forward
        m.eval()
        with QueryCounter(m) as qc0:
            model_output = m(batch_tensor)
        clean_out, clean_feat = (
            model_output if isinstance(model_output, tuple) else (model_output, None)
        )
        self.query_count += 1 + qc0.extra

        # Resolve & configure strategy for fixed perturbation
        perturb_obj = Strategy.resolve(p["strategy"])
        try:
            perturb_obj.configure(mean=ctx["mean"], std=ctx["std"], seed=ctx["seed"])
        except AttributeError:
            # Strategy doesn't have configure method, use fallback
            logger.debug(
                f"Strategy {perturb_obj.__class__.__name__} doesn't have "
                f"configure method, using fallback"
            )
            self._configure_strategy(perturb_obj, ctx["mean"], ctx["std"], None)
        except TypeError as e:
            logger.warning(f"Strategy configure() has incompatible signature: {e}")
            self._configure_strategy(perturb_obj, ctx["mean"], ctx["std"], None)
        except Exception as e:
            logger.error(f"Unexpected error configuring strategy: {e}")
            self._configure_strategy(perturb_obj, ctx["mean"], ctx["std"], None)

        # Generate perturbed batch and forward once
        with QueryCounter(m) as qc1:
            pert_tensor = perturb_obj.generate(batch_tensor, model=m)
            pert_model_output = m(pert_tensor)
        pert_out, pert_feat = (
            pert_model_output if isinstance(pert_model_output, tuple) else (pert_model_output, None)
        )
        self.query_count += 1 + qc1.extra

        # Unwrap to logits for user property & all downstream metrics
        clean_logits = self._to_logits(clean_out)
        pert_logits = self._to_logits(pert_out)

        passed_samples = self._evaluate_property_logits(
            clean_logits, pert_logits, p.get("vectorized", False), clean_feat, pert_feat
        )

        original_img, perturbed_img = self._create_image_data_pair(
            batch_tensor, clean_logits, pert_tensor, pert_logits
        )
        module_name = os.environ.get("VISPROBE_MODULE_NAME", self.user_func.__module__)

        # Analyses (use logits)
        ensemble = run_ensemble_analysis(self, clean_feat, pert_feat)
        run_meta = self._build_run_meta(perturb_obj, strength_level=None)
        per_sample = self._compute_per_sample_given(clean_logits, pert_logits, p.get("top_k"))

        return Report(
            test_name=f"{module_name}.{self.user_func.__name__}",
            test_type="given",
            runtime=time.perf_counter() - self.start_time,
            model_queries=self.query_count,
            total_samples=batch_size,
            passed_samples=passed_samples,
            original_image=original_img,
            perturbed_image=perturbed_img,
            ensemble_analysis=ensemble,
            resolution_impact=run_resolution_impact(
                self, p.get("resolutions"), perturb_obj, (clean_logits, clean_feat)
            ),
            noise_sweep_results=run_noise_sweep(
                self, p.get("noise_sweep"), (clean_logits, clean_feat)
            ),
            corruption_sweep_results=run_corruption_sweep(
                self,
                p.get("corruptions", ["gaussian_noise", "brightness", "contrast"]),
                (clean_logits, clean_feat),
            ),
            top_k_analysis=run_top_k_analysis(self, p.get("top_k"), clean_logits, pert_logits),
            perturbation_info=self._build_perturbation_info(perturb_obj),
            model_name=(
                type(ctx["model"].__dict__.get("model", ctx["model"])).__name__
                if isinstance(ctx["model"], _ModelWithIntermediateOutput)
                else type(ctx["model"]).__name__
            ),
            property_name=getattr(self.user_func, "_visprobe_property_name", None),
            strategy=f"{perturb_obj.__class__.__name__}",
            metrics={
                "layer_cosine": ensemble,
                "topk_overlap": run_top_k_analysis(self, p.get("top_k"), clean_logits, pert_logits),
            },
            runtime_sec=time.perf_counter() - self.start_time,
            num_queries=self.query_count,
            seed=ctx.get("seed"),
            run_meta=run_meta,
            per_sample=per_sample,
        )

    def _run_search(self) -> Report:
        """Runs a search for the minimal perturbation to cause a failure."""
        ctx, p = self.context, self.params
        m, batch_tensor = ctx["model"], ctx["batch_tensor"]

        m.eval()
        with QueryCounter(m) as qc0:
            model_output = m(batch_tensor)
        clean_out, clean_feat = (
            model_output if isinstance(model_output, tuple) else (model_output, None)
        )
        self.query_count += 1 + qc0.extra
        clean_logits = self._to_logits(clean_out)

        mode = self.params.get("mode", "adaptive")
        if mode == "grid":
            search_results = perform_grid_search(self, p, (clean_logits, clean_feat))
        elif mode == "random":
            search_results = perform_random_search(self, p, (clean_logits, clean_feat))
        elif mode == "binary":
            search_results = perform_binary_search(self, p, (clean_logits, clean_feat))
        else:
            search_results = perform_adaptive_search(self, p, (clean_logits, clean_feat))
        fail_thresh = search_results.get("failure_threshold")

        final_perturb_obj = (
            build_final_strategy(self, p["strategy"], fail_thresh)
            if fail_thresh is not None
            else None
        )
        if final_perturb_obj is not None:
            try:
                self._configure_strategy(
                    final_perturb_obj,
                    self.context["mean"],
                    self.context["std"],
                    self.context["rng"],
                )
            except Exception as e:
                logger.warning(f"Could not configure final strategy: {e}")

        original_img, perturbed_img, residual_panel, residual_metrics, fail_idx = build_visuals(
            self, batch_tensor, clean_logits, search_results
        )

        module_name = os.environ.get("VISPROBE_MODULE_NAME", self.user_func.__module__)

        ensemble = run_ensemble_analysis(self, clean_feat, search_results.get("perturbed_features"))

        # Paper-aligned search/metrics/aggregates blocks
        search_block, metrics_block, aggregates_block = build_search_blocks(
            self, mode, p, search_results, ensemble
        )

        run_meta = self._build_run_meta(final_perturb_obj, strength_level=fail_thresh)

        # Per-sample for search: use last step perturbed vs clean
        per_sample = []
        if search_results.get("perturbed_output") is not None:
            per_sample = self._compute_per_sample_given(
                clean_logits, search_results.get("perturbed_output"), p.get("top_k")
            )
            # add trace and threshold per sample (batch assumed homogeneous level)
            for sample in per_sample:
                sample["threshold_estimate"] = fail_thresh
                sample["trace"] = search_results.get("path")

        return Report(
            test_name=f"{module_name}.{self.user_func.__name__}",
            test_type="search",
            runtime=time.perf_counter() - self.start_time,
            model_queries=self.query_count,
            failure_threshold=fail_thresh,
            search_path=search_results.get("path"),
            original_image=original_img,
            perturbed_image=perturbed_img,
            residual_image=residual_panel,
            residual_metrics=residual_metrics,
            ensemble_analysis=ensemble,
            resolution_impact=run_resolution_impact(
                self, p.get("resolutions"), final_perturb_obj, (clean_out, clean_feat)
            ),
            noise_sweep_results=run_noise_sweep(
                self, p.get("noise_sweep"), (clean_out, clean_feat)
            ),
            corruption_sweep_results=run_corruption_sweep(
                self,
                p.get("corruptions", ["gaussian_noise", "brightness", "contrast"]),
                (clean_out, clean_feat),
            ),
            top_k_analysis=search_results.get("top_k_path"),
            perturbation_info=(
                self._build_perturbation_info(final_perturb_obj) if final_perturb_obj else None
            ),
            model_name=(
                type(self.context["model"].__dict__.get("model", self.context["model"])).__name__
                if isinstance(self.context["model"], _ModelWithIntermediateOutput)
                else type(self.context["model"]).__name__
            ),
            property_name=getattr(self.user_func, "_visprobe_property_name", None),
            strategy=(
                f"{final_perturb_obj.__class__.__name__}(eps)"
                if hasattr(final_perturb_obj, "eps")
                else f"{final_perturb_obj.__class__.__name__}"
            ),
            search=search_block,
            metrics=metrics_block,
            aggregates=aggregates_block,
            runtime_sec=time.perf_counter() - self.start_time,
            num_queries=self.query_count,
            seed=None,
            run_meta=run_meta,
            per_sample=per_sample,
        )

    # --- Small helpers to reduce complexity in _run_search ---
    def _select_first_failing_index(
        self, clean_out: torch.Tensor, pert_out: Optional[torch.Tensor], default_index: int = 0
    ) -> int:
        """Returns index of the first sample where labels differ; falls back to default_index."""
        if pert_out is None:
            return default_index
        try:
            if pert_out.dim() == 1:
                pert_out = pert_out.unsqueeze(0)
            clean_pred = torch.argmax(clean_out, dim=1)
            pert_pred = torch.argmax(pert_out, dim=1)
            mismatches = (clean_pred != pert_pred).nonzero(as_tuple=False).view(-1)
            if mismatches.numel() > 0:
                return int(mismatches[0].item())
        except Exception:
            pass
        return default_index

    def _build_residual_panel_from_batches(
        self, clean_batch: torch.Tensor, adv_batch: Optional[torch.Tensor], index: int
    ):
        """Builds a residual panel and metrics for a given sample index from two batches.

        Returns (PanelImage|None, metrics_dict|None).
        """
        if adv_batch is None:
            return None, None
        try:
            clean_px = self._denorm(clean_batch[index : index + 1])  # [1,3,H,W] in [0,1]
            adv_px = self._denorm(adv_batch[index : index + 1])

            diff = (adv_px - clean_px).float()
            linf_norm = diff.abs().max().item()
            l2_norm = diff.norm().item()

            q = 0.995
            a = torch.quantile(diff.abs().flatten(), q).item()
            a = max(a, 1e-8)
            signed = (diff / (2.0 * a) + 0.5).clamp(0.0, 1.0)

            caption = (
                f"Residual r = x_adv - x_clean (signed, q={q:.3f}). "
                f"Display scaled for visibility; L∞={linf_norm:.4f}, L2={l2_norm:.4f}"
            )
            panel = PanelImage.from_tensor(signed, caption=caption)
            metrics = {"linf_norm": linf_norm, "l2_norm": l2_norm, "scaling_factor": a}
            return panel, metrics
        except Exception:
            return None, None

    def _compute_threshold_quantiles(self, per_sample_thresholds: Optional[List[Optional[float]]]):
        """Computes p05/median/p95 quantiles over non-null per-image thresholds."""
        try:
            import numpy as _np

            per_th = [t for t in (per_sample_thresholds or []) if t is not None]
            if not per_th:
                return None
            q05, q50, q95 = _np.quantile(per_th, [0.05, 0.5, 0.95]).tolist()
            return {
                "threshold_quantiles": {"p05": float(q05), "median": float(q50), "p95": float(q95)}
            }
        except Exception:
            return None

    def _build_perturbation_info(self, strategy_obj):
        """Creates a JSON-safe PerturbationInfo for a strategy or composite strategies."""
        try:
            name, params = self._serialize_strategy(strategy_obj)
            return PerturbationInfo(name=name, params=params)
        except Exception:
            # Fallback: best-effort
            try:
                return PerturbationInfo(name=type(strategy_obj).__name__, params={})
            except Exception:
                return None

    def _serialize_strategy(self, strategy_obj):  # noqa: C901
        """Returns (name, params_dict) JSON-safe. Handles composite strategies recursively."""
        name = type(strategy_obj).__name__

        # CompositeStrategy detection by attribute
        if hasattr(strategy_obj, "strategies") and isinstance(
            getattr(strategy_obj, "strategies"), (list, tuple)
        ):
            components = []
            for s in getattr(strategy_obj, "strategies"):
                try:
                    cn, cp = self._serialize_strategy(s)
                    components.append({"name": cn, "params": cp})
                except Exception:
                    components.append({"name": type(s).__name__, "params": {}})
            return name, {"components": components}

        # Standard strategy: filter __dict__ to JSON-serializable values
        raw = {}
        try:
            raw = dict(vars(strategy_obj))
        except Exception:
            raw = {}

        def to_safe(v):
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, (list, tuple)):
                return [to_safe(x) for x in v]
            if isinstance(v, dict):
                return {str(k): to_safe(val) for k, val in v.items()}
            # For anything else (e.g., tensors, modules, callables), stringify class name
            try:
                import torch

                if isinstance(v, torch.Tensor):
                    return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
            except Exception:
                pass
            if callable(v):
                return getattr(v, "__name__", "callable")
            return str(type(v).__name__)

        params = {k: to_safe(v) for k, v in raw.items() if not k.startswith("_")}
        return name, params

    # Removed legacy _perform_search_loop and strategy resolver;
    # handled in search_modes.perform_adaptive_search

    def _evaluate_property(
        self, clean_out: torch.Tensor, pert_out: torch.Tensor, vectorized: bool
    ) -> int:
        """Calls the user's test function with logits only and returns number of passed samples."""
        batch_size = self.context["batch_size"]
        if vectorized:
            try:
                self.user_func({"output": clean_out}, {"output": pert_out})
                return batch_size
            except AssertionError:
                return 0
        passed_samples = 0

        for i in range(batch_size):
            try:
                self.user_func({"output": clean_out[i : i + 1]}, {"output": pert_out[i : i + 1]})
                passed_samples += 1
            except AssertionError:
                pass
        return passed_samples

    def _evaluate_property_mask_from_results(
        self, clean_results: Tuple, pert_results: Tuple
    ) -> List[bool]:
        """Returns per-sample pass/fail mask using logits-only inputs to the user function."""
        batch_size = self.context["batch_size"]
        clean_out, _ = clean_results
        pert_out, _ = pert_results
        mask: List[bool] = []
        for i in range(batch_size):
            try:
                self.user_func({"output": clean_out[i : i + 1]}, {"output": pert_out[i : i + 1]})
                mask.append(True)
            except AssertionError:
                mask.append(False)
        return mask

    def _create_image_data_pair(
        self, clean_tensor, clean_logits, pert_tensor, pert_logits, index: int = 0
    ):
        if clean_tensor is None or clean_logits is None:
            return None, None
        ctx = self.context
        original_image = ImageData.from_tensors(
            tensor=clean_tensor[index],
            output=clean_logits[index],
            class_names=ctx["class_names"],
            mean=ctx["mean"],
            std=ctx["std"],
        )
        if pert_tensor is None or pert_logits is None:
            return original_image, None
        perturbed_image = ImageData.from_tensors(
            tensor=pert_tensor[index],
            output=pert_logits[index],
            class_names=ctx["class_names"],
            mean=ctx["mean"],
            std=ctx["std"],
        )
        return original_image, perturbed_image

    # --- Helper: reporting/meta utilities ---
    def _infer_strength_units(self, strategy_obj) -> str:
        name = strategy_obj.__class__.__name__.lower()
        if "gaussiannoise" in name or "noise" in name:
            return "std of Gaussian"
        if "fgsm" in name or "pgd" in name:
            return "epsilon L∞"
        if "rotate" in name:
            return "degrees"
        if "brightness" in name:
            return "brightness factor"
        return "level"

    def _build_run_meta(self, strategy_obj, strength_level: Optional[float]) -> Dict[str, Any]:
        device = self.context.get("device", str(self.context["batch_tensor"].device))
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
                .decode()
                .strip()
            )
        except Exception:
            commit_hash = None
        seed = None
        try:
            seed = torch.initial_seed()
        except Exception:
            seed = None
        module_name = os.environ.get("VISPROBE_MODULE_NAME", self.user_func.__module__)
        meta = {
            "test_name": f"{module_name}.{self.user_func.__name__}",
            "commit_hash": commit_hash,
            "seed": seed,
            "device": device,
            "torch_version": getattr(_torch_version_probe, "__version__", None),
            "torchvision_version": (
                getattr(_torchvision_version_probe, "__version__", None)
                if _torchvision_version_probe
                else None
            ),
            "strategy_name": strategy_obj.__class__.__name__ if strategy_obj else None,
            "strength_units": self._infer_strength_units(strategy_obj) if strategy_obj else None,
            "strength_level": strength_level,
        }
        return meta

    def _compute_per_sample_given(
        self, clean_out: torch.Tensor, pert_out: torch.Tensor, top_k: Optional[int]
    ) -> List[Dict[str, Any]]:
        per_sample: List[Dict[str, Any]] = []
        if clean_out.dim() == 1:
            clean_out = clean_out.unsqueeze(0)
        if pert_out.dim() == 1:
            pert_out = pert_out.unsqueeze(0)
        batch = clean_out.shape[0]
        for i in range(batch):
            clean_probs = torch.softmax(clean_out[i], dim=0)
            pert_probs = torch.softmax(pert_out[i], dim=0)
            clean_label = int(torch.argmax(clean_probs).item())
            pert_label = int(torch.argmax(pert_probs).item())
            conf_drop = float(torch.max(clean_probs).item() - torch.max(pert_probs).item())
            sample: Dict[str, Any] = {
                "index": i,
                "passed": bool(clean_label == pert_label),
                "threshold_estimate": None,
                "queries": None,
                "topk_overlap": None,
                "confidence_drop": conf_drop,
                "clean_label": (
                    self.context["class_names"][clean_label]
                    if self.context["class_names"]
                    else str(clean_label)
                ),
                "pert_label": (
                    self.context["class_names"][pert_label]
                    if self.context["class_names"]
                    else str(pert_label)
                ),
                "trace": None,
            }
            if top_k and top_k > 0:
                k = min(top_k, clean_probs.shape[0])
                _, c_top = torch.topk(clean_probs, k)
                _, p_top = torch.topk(pert_probs, k)
                sample["topk_overlap"] = int(
                    len(set(c_top.tolist()).intersection(set(p_top.tolist())))
                )
            per_sample.append(sample)
        return per_sample

    def _evaluate_property_logits(
        self,
        clean_logits: torch.Tensor,
        pert_logits: torch.Tensor,
        vectorized: bool,
        clean_feat=None,
        pert_feat=None,
    ) -> int:
        """Call the user's test function with logits tensors (not tuples)."""
        batch_size = self.context["batch_size"]
        if vectorized:
            try:
                self.user_func({"output": clean_logits}, {"output": pert_logits})
                return batch_size
            except AssertionError:
                return 0

        passed = 0
        if clean_logits.dim() == 1:
            clean_logits = clean_logits.unsqueeze(0)
        if pert_logits.dim() == 1:
            pert_logits = pert_logits.unsqueeze(0)

        for i in range(batch_size):
            try:
                self.user_func(
                    {"output": clean_logits[i : i + 1]}, {"output": pert_logits[i : i + 1]}
                )
                passed += 1
            except AssertionError:
                pass
        return passed

    def _evaluate_property_mask(
        self, clean_logits: torch.Tensor, pert_logits: torch.Tensor
    ) -> List[bool]:
        """Per-sample pass/fail mask using logits tensors."""
        if clean_logits.dim() == 1:
            clean_logits = clean_logits.unsqueeze(0)
        if pert_logits.dim() == 1:
            pert_logits = pert_logits.unsqueeze(0)
        mask: List[bool] = []
        for i in range(clean_logits.shape[0]):
            try:
                self.user_func(
                    {"output": clean_logits[i : i + 1]}, {"output": pert_logits[i : i + 1]}
                )
                mask.append(True)
            except AssertionError:
                mask.append(False)
        return mask

    @staticmethod
    def _to_logits(out):
        """Accept logits or (logits, features) and return logits tensor."""
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _configure_strategy(strategy, mean, std, rng):
        """Inject normalization stats and deterministic RNG into a strategy."""
        if hasattr(strategy, "configure"):
            strategy.configure(mean=mean, std=std, generator=rng)
        else:
            if hasattr(strategy, "mean"):
                strategy.mean = mean
            if hasattr(strategy, "std"):
                strategy.std = std
            if hasattr(strategy, "_rng"):
                strategy._rng = rng

    # --- Helper: normalization utilities ---
    def _denorm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts a normalized tensor back to pixel space (0‒1).

        This is a convenience wrapper around utils.to_image_space().
        """
        from .utils import to_image_space
        return to_image_space(tensor, self.context["mean"], self.context["std"])

    def _renorm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a pixel-space tensor using the context mean/std.

        This is a convenience wrapper around utils.to_model_space().
        """
        from .utils import to_model_space
        return to_model_space(tensor, self.context["mean"], self.context["std"])
