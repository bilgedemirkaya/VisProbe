"""
This module contains framework-agnostic properties for evaluating model
robustness, primarily focused on classification tasks.
"""

from typing import Any, Literal

import torch
import torch.nn.functional as F

from .base import Property
from .helpers import extract_logits, get_topk_predictions

Mode = Literal["overlap", "containment", "jaccard"]


class LabelConstant(Property):
    """
    Asserts that the top-1 predicted label remains the same after perturbation.

    Works with batched logits and supports the tuple form (logits, features).
    Returns True only if all samples preserve the top-1 label.
    """

    def __call__(self, original: Any, perturbed: Any) -> bool:
        """
        Check if top-1 predictions are identical for all samples.

        Args:
            original: Original model outputs
            perturbed: Perturbed model outputs

        Returns:
            True if all predictions match, False otherwise
        """
        original_output = extract_logits(original)
        perturbed_output = extract_logits(perturbed)
        original_labels = torch.argmax(original_output, dim=-1)
        perturbed_labels = torch.argmax(perturbed_output, dim=-1)
        return bool(torch.all(original_labels == perturbed_labels).item())


class TopKStability(Property):
    """
    A unified top-k property with multiple modes:

    - overlap: require at least `min_overlap` common classes in top-k sets
    - containment: original top-1 must be contained in perturbed top-k
    - jaccard: Jaccard index between sets must be >= `min_jaccard`
    """

    def __init__(
        self,
        k: int = 5,
        mode: Mode = "overlap",
        *,
        min_overlap: int = 3,
        require_containment: bool = True,
        min_jaccard: float = 0.4,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        if mode == "overlap" and not 1 <= min_overlap <= k:
            raise ValueError("min_overlap must be between 1 and k.")
        if mode == "jaccard" and not 0.0 <= min_jaccard <= 1.0:
            raise ValueError("min_jaccard must be in [0,1].")

        self.k = k
        self.mode = mode
        self.min_overlap = min_overlap
        self.require_containment = require_containment
        self.min_jaccard = min_jaccard

    def __call__(self, original: Any, perturbed: Any) -> bool:
        """
        Check if top-k predictions satisfy the stability criterion.

        Args:
            original: Original model outputs
            perturbed: Perturbed model outputs

        Returns:
            True if the stability condition is met, False otherwise
        """
        original_indices, _ = get_topk_predictions(extract_logits(original), self.k)
        perturbed_indices, _ = get_topk_predictions(extract_logits(perturbed), self.k)
        original_set, perturbed_set = set(original_indices), set(perturbed_indices)

        if self.mode == "overlap":
            overlap = len(original_set & perturbed_set)
            return overlap >= self.min_overlap

        if self.mode == "containment":
            top1_original = original_indices[0]
            return (top1_original in perturbed_set) if self.require_containment else True

        if self.mode == "jaccard":
            intersection = len(original_set & perturbed_set)
            union = len(original_set | perturbed_set)
            jaccard_index = intersection / union if union else 1.0
            return jaccard_index >= self.min_jaccard

        raise ValueError(f"Unknown mode: {self.mode}")

    def __str__(self) -> str:
        base = f"{self.__class__.__name__}(k={self.k}, mode={self.mode}"
        if self.mode == "overlap":
            base += f", min_overlap={self.min_overlap})"
        elif self.mode == "containment":
            base += f", require_containment={self.require_containment})"
        else:
            base += f", min_jaccard={self.min_jaccard})"
        return base


class ConfidenceDrop(Property):
    """
    Asserts that the model's confidence in its top prediction does not drop
    by more than a specified threshold.
    """

    def __init__(self, max_drop: float = 0.3):
        if not 0.0 <= max_drop <= 1.0:
            raise ValueError("max_drop must be between 0.0 and 1.0.")
        self.max_drop = max_drop

    def __call__(self, original: Any, perturbed: Any) -> bool:
        """
        Checks if the confidence drop is within the allowed maximum for all samples.
        Uses softmax over logits to compute top-1 confidences in a batched manner.
        """
        original_output = extract_logits(original)
        perturbed_output = extract_logits(perturbed)
        original_confidences = torch.max(F.softmax(original_output, dim=-1), dim=-1).values
        perturbed_confidences = torch.max(F.softmax(perturbed_output, dim=-1), dim=-1).values
        return torch.all((original_confidences - perturbed_confidences) <= self.max_drop).item()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(max_drop={self.max_drop})"


class L2Distance(Property):
    """
    Asserts that the L2 distance between the model's output logits does not
    exceed a specified maximum delta.
    """

    def __init__(self, max_delta: float = 1.0):
        self.max_delta = max_delta

    def __call__(self, original: Any, perturbed: Any) -> bool:
        """
        Checks if the L2 distance between output vectors is within the threshold.
        """
        original_logits = extract_logits(original)
        perturbed_logits = extract_logits(perturbed)
        distance = torch.norm(original_logits - perturbed_logits, p=2).item()
        return distance <= self.max_delta

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(max_delta={self.max_delta})"
