"""
Lighting perturbations for robustness testing.

Provides various lighting/color adjustments:
- Brightness
- Contrast
- Gamma correction
- Low light simulation
- Saturation
"""

from __future__ import annotations

from typing import Any

import torch

from .base import BatchedPerturbation

__all__ = [
    "Brightness",
    "Contrast",
    "Gamma",
    "LowLight",
    "Saturation",
]


class Brightness(BatchedPerturbation):
    """
    Brightness adjustment perturbation.

    Adjusts image brightness by multiplying pixel values.

    Example:
        >>> brightness = Brightness()
        >>> darker = brightness(images, level=0.5)   # 50% brightness
        >>> brighter = brightness(images, level=1.5) # 150% brightness
    """

    name = "brightness"

    def __init__(self, clamp: bool = True) -> None:
        super().__init__()
        self.clamp = clamp

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply brightness adjustment.

        Args:
            images: Input images (N, C, H, W) or (C, H, W)
            level: Brightness factor (1.0 = unchanged, <1 = darker, >1 = brighter)

        Returns:
            Brightness-adjusted images
        """
        if level == 1.0:
            return images

        images, was_3d = self._ensure_4d(images)
        output = images * level

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def _apply_batch(
        self,
        images: torch.Tensor,
        levels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply different brightness levels per image."""
        levels_expanded = levels.view(-1, 1, 1, 1)
        output = images * levels_expanded

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return output

    def __repr__(self) -> str:
        return f"Brightness(clamp={self.clamp})"


class Contrast(BatchedPerturbation):
    """
    Contrast adjustment perturbation.

    Adjusts contrast by scaling deviation from mean.

    Example:
        >>> contrast = Contrast()
        >>> low_contrast = contrast(images, level=0.5)  # 50% contrast
        >>> high_contrast = contrast(images, level=2.0) # 200% contrast
    """

    name = "contrast"

    def __init__(self, clamp: bool = True) -> None:
        super().__init__()
        self.clamp = clamp

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply contrast adjustment.

        Args:
            images: Input images (N, C, H, W) or (C, H, W)
            level: Contrast factor (1.0 = unchanged, <1 = less contrast, >1 = more)

        Returns:
            Contrast-adjusted images
        """
        if level == 1.0:
            return images

        images, was_3d = self._ensure_4d(images)

        # Compute mean per image (keep channel dimension for broadcasting)
        mean = images.mean(dim=(-2, -1), keepdim=True)

        # Adjust contrast: output = mean + level * (input - mean)
        output = mean + level * (images - mean)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def _apply_batch(
        self,
        images: torch.Tensor,
        levels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply different contrast levels per image."""
        levels_expanded = levels.view(-1, 1, 1, 1)
        mean = images.mean(dim=(-2, -1), keepdim=True)
        output = mean + levels_expanded * (images - mean)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return output

    def __repr__(self) -> str:
        return f"Contrast(clamp={self.clamp})"


class Gamma(BatchedPerturbation):
    """
    Gamma correction perturbation.

    Applies power-law transformation: output = input^gamma

    Args:
        gain: Multiplier applied before gamma correction (default: 1.0)

    Example:
        >>> gamma = Gamma()
        >>> lighter = gamma(images, level=0.5)  # gamma < 1 brightens
        >>> darker = gamma(images, level=2.0)   # gamma > 1 darkens
    """

    name = "gamma"

    def __init__(self, gain: float = 1.0, clamp: bool = True) -> None:
        super().__init__()
        self.gain = gain
        self.clamp = clamp

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply gamma correction.

        Args:
            images: Input images (N, C, H, W) or (C, H, W)
            level: Gamma value (1.0 = unchanged)

        Returns:
            Gamma-corrected images
        """
        if level == 1.0 and self.gain == 1.0:
            return images

        images, was_3d = self._ensure_4d(images)

        # Ensure non-negative for power operation
        images_clamped = torch.clamp(images, 0.0, None)

        # Apply gamma: output = gain * input^gamma
        output = self.gain * torch.pow(images_clamped + 1e-8, level)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"Gamma(gain={self.gain}, clamp={self.clamp})"


class LowLight(BatchedPerturbation):
    """
    Low-light simulation perturbation.

    Combines reduced brightness with increased noise to simulate
    low-light conditions.

    Args:
        noise_factor: How much noise to add relative to darkening (default: 0.5)
        seed: Random seed for reproducibility

    Example:
        >>> lowlight = LowLight(noise_factor=0.3)
        >>> dim = lowlight(images, level=0.3)  # 30% brightness with noise
    """

    name = "low_light"

    def __init__(
        self,
        noise_factor: float = 0.5,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.noise_factor = noise_factor
        self.seed = seed

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply low-light simulation.

        Args:
            images: Input images (N, C, H, W) or (C, H, W)
            level: Brightness factor (0.0-1.0, lower = darker)

        Returns:
            Low-light simulated images
        """
        if level >= 1.0:
            return images

        images, was_3d = self._ensure_4d(images)

        # Reduce brightness
        output = images * level

        # Add noise proportional to darkening
        noise_std = self.noise_factor * (1.0 - level)
        if noise_std > 0:
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=images.device)
                generator.manual_seed(self.seed)

            noise = torch.randn(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=generator,
            )
            output = output + noise_std * noise

        output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def __repr__(self) -> str:
        return f"LowLight(noise_factor={self.noise_factor}, seed={self.seed})"


class Saturation(BatchedPerturbation):
    """
    Saturation adjustment perturbation.

    Adjusts color saturation (0 = grayscale, 1 = original, >1 = oversaturated).

    Example:
        >>> sat = Saturation()
        >>> gray = sat(images, level=0.0)     # Grayscale
        >>> vivid = sat(images, level=2.0)    # Double saturation
    """

    name = "saturation"

    def __init__(self, clamp: bool = True) -> None:
        super().__init__()
        self.clamp = clamp

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply saturation adjustment.

        Args:
            images: Input images (N, C, H, W) or (C, H, W) - assumes RGB
            level: Saturation factor (0 = grayscale, 1 = unchanged)

        Returns:
            Saturation-adjusted images
        """
        if level == 1.0:
            return images

        images, was_3d = self._ensure_4d(images)

        # Convert to grayscale using luminance weights
        # ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
        weights = weights.view(1, 3, 1, 1)

        gray = (images * weights).sum(dim=1, keepdim=True)
        gray = gray.expand_as(images)

        # Interpolate between grayscale and original
        output = gray + level * (images - gray)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return self._restore_dims(output, was_3d)

    def _apply_batch(
        self,
        images: torch.Tensor,
        levels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply different saturation levels per image."""
        weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
        weights = weights.view(1, 3, 1, 1)

        gray = (images * weights).sum(dim=1, keepdim=True)
        gray = gray.expand_as(images)

        levels_expanded = levels.view(-1, 1, 1, 1)
        output = gray + levels_expanded * (images - gray)

        if self.clamp:
            output = torch.clamp(output, 0.0, 1.0)

        return output

    def __repr__(self) -> str:
        return f"Saturation(clamp={self.clamp})"
