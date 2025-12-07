"""
Defines the data structures for VisProbe test reports.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as T

__all__ = ["ImageData", "Report", "PerturbationInfo"]


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Removes directory separators and other potentially dangerous characters,
    keeping only alphanumeric characters, underscores, hyphens, and periods.

    Args:
        name: The filename to sanitize

    Returns:
        A safe filename string
    """
    # Remove any path components (e.g., "../", "/etc/", etc.)
    name = os.path.basename(name)
    # Replace any remaining potentially dangerous characters
    # Keep only alphanumeric, underscore, hyphen, and period
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    # Prevent empty strings or strings that are only periods
    if not name or name.strip('.') == '':
        name = 'unnamed_test'
    return name


def get_results_dir() -> str:
    """
    Get the platform-appropriate results directory.

    Priority:
    1. VISPROBE_RESULTS_DIR environment variable
    2. System temp directory + 'visprobe_results'

    Returns:
        str: Absolute path to the results directory
    """
    env_dir = os.environ.get("VISPROBE_RESULTS_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.join(tempfile.gettempdir(), "visprobe_results")


@dataclass
class PanelImage:
    """Represents a generic image panel for diagnostics (no prediction needed)."""

    image_b64: str
    caption: str

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        caption: str = "",
    ) -> "PanelImage":
        """
        Create a PanelImage from a tensor.

        Args:
            tensor: Image tensor to convert
            mean: Optional channel means for denormalization
            std: Optional channel stds for denormalization
            caption: Caption text for the image

        Returns:
            PanelImage instance with base64-encoded image
        """
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            tensor = tensor.detach().cpu() * std_t + mean_t
        else:
            tensor = tensor.detach().cpu()
        img_chw = tensor.squeeze(0).clamp(0, 1)
        pil_image = T.ToPILImage()(img_chw)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return cls(image_b64=image_b64, caption=caption)


@dataclass
class PerturbationInfo:
    """Contains metadata about the perturbation applied."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


from .utils import NumpyEncoder  # noqa: E402


@dataclass
class ImageData:
    """Represents an image and its model prediction for visualization."""

    image_b64: str
    prediction: str
    confidence: float

    @classmethod
    def from_tensors(
        cls,
        tensor: torch.Tensor,
        output: torch.Tensor,
        class_names: Optional[List[str]],
        mean: Optional[List[float]],
        std: Optional[List[float]],
    ) -> "ImageData":
        """Creates an ImageData object from tensors."""
        if mean and std:
            mean_t = torch.tensor(mean).view(3, 1, 1)
            std_t = torch.tensor(std).view(3, 1, 1)
            tensor = tensor.detach().cpu() * std_t + mean_t

        # Ensure shape [C,H,W] and avoid unnecessary resampling
        img_chw = tensor.squeeze(0).clamp(0, 1)
        pil_image = T.ToPILImage()(img_chw)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        probabilities = torch.nn.functional.softmax(output, dim=0)
        confidence, pred_idx = torch.max(probabilities, 0)
        prediction_label = (
            class_names[pred_idx.item()] if class_names else f"Class {pred_idx.item()}"
        )

        return cls(image_b64=image_b64, prediction=prediction_label, confidence=confidence.item())


@dataclass
class Report:
    """A unified report for all VisProbe test types."""

    test_name: str
    test_type: str
    runtime: float
    model_queries: int
    # Paper-aligned summary fields (optional and additive for compatibility)
    model_name: Optional[str] = None
    dataset: Optional[str] = None
    property_name: Optional[str] = None
    strategy: Optional[str] = None
    search: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    runtime_sec: Optional[float] = None
    num_queries: Optional[int] = None
    seed: Optional[int] = None
    original_image: Optional[ImageData] = None
    perturbed_image: Optional[ImageData] = None
    ensemble_analysis: Optional[Dict[str, Any]] = None
    resolution_impact: Optional[Dict[str, float]] = None
    noise_sweep_results: Optional[List[Dict[str, Any]]] = None
    corruption_sweep_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    total_samples: Optional[int] = None
    passed_samples: Optional[int] = None
    failure_threshold: Optional[float] = None
    search_path: Optional[List[Dict[str, Any]]] = None
    perturbation_info: Optional[PerturbationInfo] = None
    top_k_analysis: Any = None  # Can be an int for 'given' or list for 'search'
    residual_image: Optional[PanelImage] = None
    residual_metrics: Optional[Dict[str, float]] = None
    # Extended, paper-aligned structures
    run_meta: Optional[Dict[str, Any]] = None
    per_sample: Optional[List[Dict[str, Any]]] = None
    aggregates: Optional[Dict[str, Any]] = None

    @property
    def robust_accuracy(self) -> Optional[float]:
        if (
            self.test_type == "given"
            and self.total_samples is not None
            and self.passed_samples is not None
        ):
            if self.total_samples == 0:
                return 1.0
            return self.passed_samples / self.total_samples
        return None

    def to_json(self) -> str:
        """Serializes the report to a JSON string with images simplified (no base64)."""
        data = self._build_serializable_dict(results_dir=None)
        return json.dumps(data, cls=NumpyEncoder, indent=2)

    def save(self):
        """Saves the report to a JSON file and writes images to disk (no base64 in JSON)."""
        try:
            results_dir = get_results_dir()
            os.makedirs(results_dir, exist_ok=True)
            # Sanitize test name to prevent path traversal
            safe_name = _sanitize_filename(self.test_name)
            data = self._build_serializable_dict(results_dir=results_dir)
            file_path = os.path.join(results_dir, f"{safe_name}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
            print(f"✅ Report saved to {file_path}")
            # Also save a CSV for quick plotting if per-sample exists
            try:
                if self.per_sample:
                    csv_path = os.path.join(results_dir, f"{safe_name}.csv")
                    fieldnames = sorted({k for row in self.per_sample for k in row.keys()})
                    with open(csv_path, "w", newline="") as cf:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in self.per_sample:
                            writer.writerow(row)
            except Exception as ce:
                print(f"⚠️  Could not save CSV for '{self.test_name}': {ce}")
        except Exception as e:
            print(f"⚠️  Could not save test report for '{self.test_name}': {e}")

    def _build_serializable_dict(self, results_dir: Optional[str]) -> Dict[str, Any]:  # noqa: C901
        """
        Build a JSON-serializable dictionary for this report, removing base64 image blobs.
        If results_dir is provided, write image files to disk and include file paths instead.
        """
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if self.robust_accuracy is not None:
            data["robust_accuracy"] = self.robust_accuracy
        if "runtime_sec" not in data:
            data["runtime_sec"] = self.runtime
        if "num_queries" not in data:
            data["num_queries"] = self.model_queries
        if "property_name" in data:
            data["property"] = data.pop("property_name")

        def write_image(b64_str: str, suffix: str) -> Optional[str]:
            if not results_dir:
                return None
            try:
                img_bytes = base64.b64decode(b64_str)
                # Sanitize test name to prevent path traversal
                safe_name = _sanitize_filename(self.test_name)
                img_path = os.path.join(results_dir, f"{safe_name}.{suffix}.png")
                with open(img_path, "wb") as imf:
                    imf.write(img_bytes)
                return img_path
            except Exception:
                return None

        # Simplify image fields
        if "original_image" in data and data["original_image"]:
            oi = data["original_image"]
            img_path = write_image(oi.get("image_b64", ""), "original")
            data["original_image"] = {
                "image_path": img_path,
                "prediction": oi.get("prediction"),
                "confidence": oi.get("confidence"),
            }
        if "perturbed_image" in data and data["perturbed_image"]:
            pi = data["perturbed_image"]
            img_path = write_image(pi.get("image_b64", ""), "perturbed")
            data["perturbed_image"] = {
                "image_path": img_path,
                "prediction": pi.get("prediction"),
                "confidence": pi.get("confidence"),
            }
        if "residual_image" in data and data["residual_image"]:
            ri = data["residual_image"]
            img_path = write_image(ri.get("image_b64", ""), "residual")
            data["residual_image"] = {
                "image_path": img_path,
                "caption": ri.get("caption", "Residual"),
            }

        return data
