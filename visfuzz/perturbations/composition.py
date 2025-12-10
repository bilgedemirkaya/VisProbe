"""
Perturbation composition utilities.

Provides ways to combine multiple perturbations:
- Sequential composition (apply one after another)
- Parallel composition (apply all and blend)
- Random selection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .base import Perturbation

__all__ = [
    "Compose",
    "RandomChoice",
    "Blend",
    "LowLightBlur",
]


class Compose(Perturbation):
    """
    Sequential composition of perturbations.

    Applies perturbations one after another in order.

    Args:
        perturbations: List of perturbations to compose
        level_mode: How to distribute level across perturbations
            - 'same': Use same level for all
            - 'split': Divide level equally
            - 'first': Apply level to first only

    Example:
        >>> composed = Compose([LowLight(), GaussianBlur()])
        >>> result = composed(images, level=0.5)
    """

    name = "compose"

    def __init__(
        self,
        perturbations: Sequence[Perturbation],
        level_mode: str = "same",
    ) -> None:
        super().__init__()
        self.perturbations = list(perturbations)
        if level_mode not in ("same", "split", "first"):
            raise ValueError(f"level_mode must be 'same', 'split', or 'first', got {level_mode}")
        self.level_mode = level_mode

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply all perturbations sequentially.

        Args:
            images: Input images
            level: Perturbation intensity

        Returns:
            Images after all perturbations applied
        """
        output = images

        for i, pert in enumerate(self.perturbations):
            # Determine level for this perturbation
            if self.level_mode == "same":
                pert_level = level
            elif self.level_mode == "split":
                pert_level = level / len(self.perturbations)
            else:  # first
                pert_level = level if i == 0 else 1.0  # Identity level for others

            output = pert(output, pert_level, **kwargs)

        return output

    def __repr__(self) -> str:
        pert_names = [p.name for p in self.perturbations]
        return f"Compose({pert_names}, level_mode={self.level_mode})"


class RandomChoice(Perturbation):
    """
    Randomly select one perturbation to apply.

    Args:
        perturbations: List of perturbations to choose from
        weights: Optional weights for selection probabilities
        seed: Random seed for reproducibility

    Example:
        >>> random_pert = RandomChoice([GaussianNoise(), GaussianBlur()])
        >>> result = random_pert(images, level=0.5)
    """

    name = "random_choice"

    def __init__(
        self,
        perturbations: Sequence[Perturbation],
        weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.perturbations = list(perturbations)
        self.weights = weights
        self.seed = seed
        self._generator: Optional[torch.Generator] = None

    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        """Get random generator."""
        if self.seed is None:
            return None
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply a randomly selected perturbation.

        Args:
            images: Input images
            level: Perturbation intensity

        Returns:
            Perturbed images
        """
        # Select perturbation
        if self.weights is not None:
            weights_tensor = torch.tensor(self.weights, device=images.device)
            idx = torch.multinomial(
                weights_tensor,
                1,
                generator=self._get_generator(images.device),
            ).item()
        else:
            gen = self._get_generator(images.device)
            if gen is not None:
                idx = torch.randint(0, len(self.perturbations), (1,), generator=gen).item()
            else:
                idx = torch.randint(0, len(self.perturbations), (1,)).item()

        return self.perturbations[idx](images, level, **kwargs)

    def __repr__(self) -> str:
        pert_names = [p.name for p in self.perturbations]
        return f"RandomChoice({pert_names}, weights={self.weights})"


class Blend(Perturbation):
    """
    Blend multiple perturbations together.

    Applies all perturbations and blends results.

    Args:
        perturbations: List of perturbations to blend
        blend_weights: Weights for blending (default: equal weights)

    Example:
        >>> blended = Blend([GaussianNoise(), GaussianBlur()], blend_weights=[0.7, 0.3])
        >>> result = blended(images, level=0.5)
    """

    name = "blend"

    def __init__(
        self,
        perturbations: Sequence[Perturbation],
        blend_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.perturbations = list(perturbations)

        if blend_weights is None:
            blend_weights = [1.0 / len(perturbations)] * len(perturbations)
        elif len(blend_weights) != len(perturbations):
            raise ValueError("blend_weights must match number of perturbations")

        # Normalize weights
        total = sum(blend_weights)
        self.blend_weights = [w / total for w in blend_weights]

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply all perturbations and blend results.

        Args:
            images: Input images
            level: Perturbation intensity

        Returns:
            Blended result of all perturbations
        """
        result = torch.zeros_like(images)

        for pert, weight in zip(self.perturbations, self.blend_weights):
            perturbed = pert(images, level, **kwargs)
            result = result + weight * perturbed

        return result

    def __repr__(self) -> str:
        pert_names = [p.name for p in self.perturbations]
        return f"Blend({pert_names}, weights={self.blend_weights})"


class LowLightBlur(Perturbation):
    """
    Combined low-light and blur perturbation.

    Simulates challenging visibility conditions: dim lighting + focus blur.
    This is a common real-world scenario (e.g., night driving, indoor dim lighting).

    Args:
        noise_factor: Noise intensity in low-light simulation
        blur_kernel_size: Fixed blur kernel size (or None for auto)

    Example:
        >>> low_light_blur = LowLightBlur()
        >>> result = low_light_blur(images, level=0.5)
    """

    name = "low_light_blur"

    def __init__(
        self,
        noise_factor: float = 0.3,
        blur_kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.noise_factor = noise_factor
        self.blur_kernel_size = blur_kernel_size

        # Lazy init components
        self._low_light: Optional[Perturbation] = None
        self._blur: Optional[Perturbation] = None

    def _ensure_components(self) -> None:
        """Lazy initialization of component perturbations."""
        if self._low_light is None:
            from .lighting import LowLight
            from .blur import GaussianBlur

            self._low_light = LowLight(noise_factor=self.noise_factor)
            self._blur = GaussianBlur(kernel_size=self.blur_kernel_size)

    def apply(
        self,
        images: torch.Tensor,
        level: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply combined low-light and blur.

        Args:
            images: Input images
            level: Combined intensity (0-1)
                - Low light: brightness = 1 - level * 0.7
                - Blur: sigma = level * 3

        Returns:
            Images with low-light and blur applied
        """
        self._ensure_components()

        # Split level between effects
        brightness = max(0.1, 1.0 - level * 0.7)  # 1.0 -> 0.3
        blur_sigma = level * 3.0  # 0 -> 3

        # Apply low light first (brightness reduction + noise)
        output = self._low_light(images, brightness, **kwargs)

        # Then apply blur
        if blur_sigma > 0.1:
            output = self._blur(output, blur_sigma, **kwargs)

        return output

    def __repr__(self) -> str:
        return f"LowLightBlur(noise_factor={self.noise_factor}, blur_kernel_size={self.blur_kernel_size})"
