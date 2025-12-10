"""
Base perturbation class for visfuzz.

Provides a clean, GPU-optimized interface for image perturbations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["Perturbation", "BatchedPerturbation"]


class Perturbation(ABC):
    """
    Abstract base class for all perturbations.

    Perturbations transform input images to test model robustness.
    All perturbations support:
    - Batched operations (process multiple images at once)
    - GPU acceleration (computations stay on device)
    - Level-based intensity control

    Attributes:
        name: Human-readable name of the perturbation
    """

    name: str = "base"

    def __init__(self) -> None:
        self._device: Optional[torch.device] = None
        self._dtype: torch.dtype = torch.float32

    @abstractmethod
    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply perturbation to images at given intensity level.

        Args:
            images: Input images tensor (N, C, H, W) or (C, H, W)
            level: Perturbation intensity (0.0 = no effect, 1.0 = max effect)
            **kwargs: Additional perturbation-specific parameters

        Returns:
            Perturbed images tensor with same shape as input
        """
        raise NotImplementedError

    def __call__(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Shorthand for apply()."""
        return self.apply(images, level, **kwargs)

    def to(self, device: Union[str, torch.device]) -> "Perturbation":
        """Move perturbation to specified device."""
        self._device = torch.device(device) if isinstance(device, str) else device
        return self

    @property
    def device(self) -> Optional[torch.device]:
        """Get current device."""
        return self._device

    def _ensure_4d(self, images: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Ensure images are 4D tensor (N, C, H, W).

        Returns:
            Tuple of (4D tensor, was_3d_flag)
        """
        if images.dim() == 3:
            return images.unsqueeze(0), True
        elif images.dim() == 4:
            return images, False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {images.dim()}D")

    def _restore_dims(self, images: torch.Tensor, was_3d: bool) -> torch.Tensor:
        """Restore original dimensions if needed."""
        return images.squeeze(0) if was_3d else images

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BatchedPerturbation(Perturbation):
    """
    Base class for perturbations optimized for batch processing.

    Provides utilities for efficient GPU batch operations.
    """

    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()
        self.batch_size = batch_size

    def apply_batched(
        self,
        images: torch.Tensor,
        levels: Union[float, List[float], torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply perturbation with potentially different levels per image.

        Args:
            images: Input images (N, C, H, W)
            levels: Single level or per-image levels
            **kwargs: Additional parameters

        Returns:
            Perturbed images
        """
        images, was_3d = self._ensure_4d(images)
        n = images.shape[0]

        # Normalize levels to tensor
        if isinstance(levels, (int, float)):
            levels_tensor = torch.full((n,), levels, device=images.device, dtype=images.dtype)
        elif isinstance(levels, list):
            levels_tensor = torch.tensor(levels, device=images.device, dtype=images.dtype)
        else:
            levels_tensor = levels.to(device=images.device, dtype=images.dtype)

        # Process in batches
        results = []
        for i in range(0, n, self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_levels = levels_tensor[i:i + self.batch_size]

            # Apply perturbation to batch
            perturbed = self._apply_batch(batch, batch_levels, **kwargs)
            results.append(perturbed)

        output = torch.cat(results, dim=0)
        return self._restore_dims(output, was_3d)

    def _apply_batch(
        self,
        images: torch.Tensor,
        levels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply perturbation to a batch. Override for custom batch handling.

        Default implementation applies same level to all images.
        """
        # Default: use first level for all (subclasses can override for per-image levels)
        return self.apply(images, float(levels[0].item()), **kwargs)
