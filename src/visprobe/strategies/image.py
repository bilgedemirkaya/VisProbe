"""
Image-based perturbation strategies (natural transformations).

Provides noise, brightness, rotation and other non-adversarial perturbations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from io import BytesIO

import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image

from .base import Strategy


class NoOpStrategy(Strategy):
    """Identity strategy that returns images unchanged."""

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """Return images unchanged (identity transformation)."""
        return imgs

    def __str__(self) -> str:
        return "NoOpStrategy()"


class GaussianNoiseStrategy(Strategy):
    """
    Additive Gaussian noise in pixel space.

    Applies noise as: clip(denorm(x) + σ·N(0,1), 0, 1) then renormalizes.

    Args:
        std_dev: Noise standard deviation (in pixel space [0,1])
        mean: Channel means for denormalization (e.g., ImageNet means)
        std: Channel stds for denormalization (e.g., ImageNet stds)
        seed: Random seed for reproducibility (optional)

    Notes:
        - If mean/std are None, assumes inputs are already in [0,1] pixel space
        - Uses local RNG to avoid polluting global state
        - Works with 3D (C,H,W) and 4D (N,C,H,W) tensors
    """

    def __init__(
        self,
        std_dev: float,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if std_dev < 0:
            raise ValueError("std_dev must be non-negative")
        self.std_dev = float(std_dev)
        self.mean = tuple(mean) if mean is not None else None
        self.std = tuple(std) if std is not None else None
        self._seed = seed

        # Cache for stats tensors (per device)
        self._stats_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    def configure(
        self,
        *,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        seed: Optional[int] = None,
        **kwargs,  # Ignore unknown args for compatibility
    ) -> "GaussianNoiseStrategy":
        """Configure normalization stats and seed."""
        if mean is not None:
            self.mean = tuple(mean)
            self._stats_cache.clear()  # Invalidate cache
        if std is not None:
            self.std = tuple(std)
            self._stats_cache.clear()
        if seed is not None:
            self._seed = seed
        return self

    def _get_stats(
        self, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached mean/std tensors for device."""
        if self.mean is None or self.std is None:
            return None, None

        if device not in self._stats_cache:
            mean_t = torch.tensor(self.mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            std_t = torch.tensor(self.std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            self._stats_cache[device] = (mean_t, std_t)

        return self._stats_cache[device]

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        """Generate random noise without polluting global RNG."""
        if self._seed is not None:
            # Use local generator to avoid global state mutation
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self._seed)
            return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)
        return torch.randn_like(x)

    def generate(
        self,
        imgs: torch.Tensor,
        model: nn.Module,
        level: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply Gaussian noise to images."""
        # Handle 3D input (C,H,W) -> (1,C,H,W)
        squeeze_back = imgs.dim() == 3
        if squeeze_back:
            imgs = imgs.unsqueeze(0)
        elif imgs.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {imgs.dim()}D")

        sigma = level if level is not None else self.std_dev
        if sigma == 0.0:
            return imgs.squeeze(0) if squeeze_back else imgs

        mean, std = self._get_stats(imgs.device)
        noise = sigma * self._randn_like(imgs)

        if mean is None:
            # Already in pixel space
            out = torch.clamp(imgs + noise, 0.0, 1.0)
        else:
            # Denorm -> add noise -> clamp -> renorm
            out = imgs * std + mean  # to pixel space
            out = torch.clamp(out + noise, 0.0, 1.0)
            out = (out - mean) / std  # back to model space

        return out.squeeze(0) if squeeze_back else out

    def __str__(self) -> str:
        return f"GaussianNoiseStrategy(std_dev={self.std_dev})"

    def __repr__(self) -> str:
        return f"GaussianNoiseStrategy(std_dev={self.std_dev}, seed={self._seed})"


class BrightnessStrategy(Strategy):
    """
    Brightness adjustment perturbation.

    Args:
        brightness_factor: Multiplier for brightness (1.0 = unchanged)
    """

    def __init__(self, brightness_factor: float):
        if brightness_factor < 0.0:
            raise ValueError("brightness_factor must be non-negative")
        self.brightness_factor = float(brightness_factor)

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply brightness adjustment to images.

        Args:
            imgs: Input images
            model: Model (unused)
            level: Optional brightness factor (overrides instance value)

        Returns:
            Brightness-adjusted images
        """
        factor = level if level is not None else self.brightness_factor
        # Special case: factor=1.0 should return unchanged images
        if factor == 1.0:
            return imgs
        return F.adjust_brightness(imgs, factor)

    def __str__(self) -> str:
        return f"BrightnessStrategy(factor={self.brightness_factor})"

    def __repr__(self) -> str:
        return f"BrightnessStrategy(brightness_factor={self.brightness_factor})"


class ContrastStrategy(Strategy):
    """
    Contrast adjustment perturbation.

    Args:
        contrast_factor: Multiplier for contrast (1.0 = unchanged)
    """

    def __init__(self, contrast_factor: float):
        if contrast_factor < 0.0:
            raise ValueError("contrast_factor must be non-negative")
        self.contrast_factor = float(contrast_factor)

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply contrast adjustment to images.

        Args:
            imgs: Input images
            model: Model (unused)
            level: Optional contrast factor (overrides instance value)

        Returns:
            Contrast-adjusted images
        """
        factor = level if level is not None else self.contrast_factor
        # Special case: factor=1.0 should return unchanged images
        if factor == 1.0:
            return imgs
        return F.adjust_contrast(imgs, factor)

    def __str__(self) -> str:
        return f"ContrastStrategy(factor={self.contrast_factor})"

    def __repr__(self) -> str:
        return f"ContrastStrategy(contrast_factor={self.contrast_factor})"


class RotateStrategy(Strategy):
    """
    Rotation perturbation.

    Args:
        angle: Rotation angle in degrees (counter-clockwise)
    """

    def __init__(self, angle: float):
        self.angle = float(angle)

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply rotation to images.

        Args:
            imgs: Input images
            model: Model (unused)
            level: Optional rotation angle in degrees (overrides instance value)

        Returns:
            Rotated images
        """
        angle = level if level is not None else self.angle
        return F.rotate(imgs, angle)

    def __str__(self) -> str:
        return f"RotateStrategy(angle={self.angle})"

    def __repr__(self) -> str:
        return f"RotateStrategy(angle={self.angle})"


class GammaStrategy(Strategy):
    """
    Gamma correction perturbation.

    Adjusts image gamma (brightness in non-linear way).
    gamma < 1.0 brightens, gamma > 1.0 darkens.

    Args:
        gamma: Gamma value (1.0 = unchanged)
        gain: Multiplier applied before gamma correction (default: 1.0)
    """

    def __init__(self, gamma: float, gain: float = 1.0):
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        self.gamma = float(gamma)
        self.gain = float(gain)

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply gamma correction to images.

        Args:
            imgs: Input images
            model: Model (unused)
            level: Optional gamma value (overrides instance value)

        Returns:
            Gamma-corrected images
        """
        gamma = level if level is not None else self.gamma
        if gamma == 1.0:
            return imgs
        return F.adjust_gamma(imgs, gamma, gain=self.gain)

    def __str__(self) -> str:
        return f"GammaStrategy(gamma={self.gamma})"

    def __repr__(self) -> str:
        return f"GammaStrategy(gamma={self.gamma}, gain={self.gain})"


class GaussianBlurStrategy(Strategy):
    """
    Gaussian blur perturbation.

    Applies Gaussian blur with specified kernel size and sigma.

    Args:
        kernel_size: Size of Gaussian kernel (must be odd, or tuple of odds)
        sigma: Gaussian standard deviation (can be tuple for anisotropic blur)
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int] = 5,
        sigma: float | Tuple[float, float] = 1.0,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply Gaussian blur to images.

        Args:
            imgs: Input images
            model: Model (unused)
            level: Optional sigma value (overrides instance value)

        Returns:
            Blurred images
        """
        sigma = level if level is not None else self.sigma
        if sigma == 0.0:
            return imgs

        # Convert scalar sigma to tuple if needed
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))

        return F.gaussian_blur(imgs, self.kernel_size, sigma)

    def __str__(self) -> str:
        return f"GaussianBlurStrategy(kernel_size={self.kernel_size}, sigma={self.sigma})"

    def __repr__(self) -> str:
        return f"GaussianBlurStrategy(kernel_size={self.kernel_size}, sigma={self.sigma})"


class MotionBlurStrategy(Strategy):
    """
    Motion blur perturbation (horizontal motion).

    Simulates camera motion blur by applying a directional blur kernel.

    Args:
        kernel_size: Size of motion blur kernel (larger = more blur)
        angle: Direction of motion in degrees (0 = horizontal, 90 = vertical)
    """

    def __init__(self, kernel_size: int = 15, angle: float = 0.0):
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        self.kernel_size = kernel_size
        self.angle = float(angle)

    def _create_motion_kernel(
        self, size: int, angle: float, device: torch.device
    ) -> torch.Tensor:
        """Create a motion blur kernel."""
        # Create horizontal line kernel
        kernel = torch.zeros((size, size), device=device, dtype=torch.float32)
        mid = size // 2
        kernel[mid, :] = 1.0
        kernel = kernel / kernel.sum()

        # Rotate if angle is not 0
        if angle != 0.0:
            # Convert to PIL for rotation, then back
            kernel_pil = F.to_pil_image(kernel.unsqueeze(0))
            kernel_pil = kernel_pil.rotate(angle, resample=Image.BILINEAR)
            kernel = F.to_tensor(kernel_pil).squeeze(0).to(device)
            kernel = kernel / kernel.sum()  # Re-normalize

        return kernel

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply motion blur to images.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Model (unused)
            level: Optional kernel size (overrides instance value)

        Returns:
            Motion-blurred images
        """
        # Handle 3D input
        squeeze_back = imgs.dim() == 3
        if squeeze_back:
            imgs = imgs.unsqueeze(0)
        elif imgs.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {imgs.dim()}D")

        kernel_size = int(level) if level is not None else self.kernel_size
        if kernel_size < 2:
            return imgs.squeeze(0) if squeeze_back else imgs

        # Create motion blur kernel
        kernel = self._create_motion_kernel(kernel_size, self.angle, imgs.device)

        # Expand kernel for all channels
        # kernel shape: (1, 1, H, W) for conv2d
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # Apply convolution per channel
        N, C, H, W = imgs.shape
        blurred = []
        for c in range(C):
            channel = imgs[:, c:c+1, :, :]
            # Apply same padding to maintain size
            padding = kernel_size // 2
            blurred_channel = torch.nn.functional.conv2d(
                channel, kernel, padding=padding
            )
            blurred.append(blurred_channel)

        out = torch.cat(blurred, dim=1)
        return out.squeeze(0) if squeeze_back else out

    def __str__(self) -> str:
        return f"MotionBlurStrategy(kernel_size={self.kernel_size}, angle={self.angle})"

    def __repr__(self) -> str:
        return f"MotionBlurStrategy(kernel_size={self.kernel_size}, angle={self.angle})"


class JPEGCompressionStrategy(Strategy):
    """
    JPEG compression artifact perturbation.

    Simulates lossy JPEG compression by encoding/decoding at specified quality.

    Args:
        quality: JPEG quality level (0-100, lower = more artifacts)
    """

    def __init__(self, quality: int = 75):
        if not 0 <= quality <= 100:
            raise ValueError("quality must be in [0, 100]")
        self.quality = int(quality)

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply JPEG compression artifacts to images.

        Args:
            imgs: Input images (N, C, H, W) or (C, H, W)
            model: Model (unused)
            level: Optional quality level (overrides instance value)

        Returns:
            JPEG-compressed images
        """
        quality = int(level) if level is not None else self.quality
        if quality >= 100:
            return imgs

        # Handle 3D input
        squeeze_back = imgs.dim() == 3
        if squeeze_back:
            imgs = imgs.unsqueeze(0)
        elif imgs.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {imgs.dim()}D")

        device = imgs.device
        dtype = imgs.dtype

        # Process each image
        compressed = []
        for img_tensor in imgs:
            # Convert to PIL Image (assume normalized to [0, 1])
            img_pil = F.to_pil_image(img_tensor.cpu())

            # Encode/decode as JPEG
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)

            # Convert back to tensor
            img_tensor_compressed = F.to_tensor(img_compressed).to(device=device, dtype=dtype)
            compressed.append(img_tensor_compressed)

        out = torch.stack(compressed)
        return out.squeeze(0) if squeeze_back else out

    def __str__(self) -> str:
        return f"JPEGCompressionStrategy(quality={self.quality})"

    def __repr__(self) -> str:
        return f"JPEGCompressionStrategy(quality={self.quality})"
