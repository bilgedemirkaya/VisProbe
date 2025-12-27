"""
The `visprobe.strategies` module provides a collection of perturbation
strategies that can be used in robustness testing.

Strategies cover multiple perturbation types:
- Adversarial attacks (FGSM, PGD, BIM, APGD, Square)
- Image perturbations (Gaussian noise, brightness, contrast, rotation)
- Blur effects (Gaussian, motion, defocus, box)
- Lighting changes (brightness, contrast, gamma, low-light, saturation)
- Spatial transformations (rotation, scale, translation, shear, elastic deformation)
- Composition and blending of perturbations
"""

from __future__ import annotations

from typing import Any, Optional

from .adversarial import APGDStrategy, BIMStrategy, FGSMStrategy, PGDStrategy, SquareAttackStrategy
from .base import Strategy
from .image import BrightnessStrategy, ContrastStrategy, GaussianNoiseStrategy, RotateStrategy

# Import perturbation classes
from .blur import BoxBlur, DefocusBlur, GaussianBlur, MotionBlur
from .composition import Blend, Compose, LowLightBlur, RandomChoice
from .lighting import Brightness, Contrast, Gamma, LowLight, Saturation
from .noise import GaussianNoise, SaltPepperNoise, SpeckleNoise, UniformNoise
from .perturbation_base import Perturbation
from .spatial import ElasticDeform, Rotation, Scale, Shear, Translation


# ============================================================================
# Perturbation to Strategy Adapter
# ============================================================================


class PerturbationStrategy(Strategy):
    """
    Adapter that wraps a Perturbation object as a visprobe Strategy.

    Converts the Perturbation.apply(images, level) interface to
    Strategy.generate(imgs, model, level) interface.
    """

    def __init__(self, perturbation: Perturbation) -> None:
        """
        Initialize with a Perturbation instance.

        Args:
            perturbation: Initialized Perturbation instance
        """
        self.perturbation = perturbation

    def generate(self, imgs: Any, model: Any, level: Optional[float] = None) -> Any:
        """
        Apply the wrapped perturbation.

        Args:
            imgs: Input images to perturb
            model: The model being tested (not used)
            level: Perturbation intensity level

        Returns:
            Perturbed images
        """
        if level is None:
            level = 0.5  # Default to 50% intensity

        return self.perturbation.apply(imgs, level=level)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.perturbation})"


# ============================================================================
# Noise Perturbation Strategies
# ============================================================================


class SaltPepperNoiseStrategy(PerturbationStrategy):
    """Salt and pepper noise perturbation strategy."""

    def __init__(self, salt_ratio: float = 0.5) -> None:
        """Initialize salt and pepper noise strategy."""
        perturbation = SaltPepperNoise(salt_ratio=salt_ratio)
        super().__init__(perturbation)


class UniformNoiseStrategy(PerturbationStrategy):
    """Uniform noise perturbation strategy."""

    def __init__(self) -> None:
        """Initialize uniform noise strategy."""
        perturbation = UniformNoise()
        super().__init__(perturbation)


class SpeckleNoiseStrategy(PerturbationStrategy):
    """Speckle noise perturbation strategy."""

    def __init__(self) -> None:
        """Initialize speckle noise strategy."""
        perturbation = SpeckleNoise()
        super().__init__(perturbation)


# ============================================================================
# Blur Perturbation Strategies
# ============================================================================


class GaussianBlurStrategy(PerturbationStrategy):
    """Gaussian blur perturbation strategy."""

    def __init__(self) -> None:
        """Initialize Gaussian blur strategy."""
        perturbation = GaussianBlur()
        super().__init__(perturbation)


class MotionBlurStrategy(PerturbationStrategy):
    """Motion blur perturbation strategy."""

    def __init__(self) -> None:
        """Initialize motion blur strategy."""
        perturbation = MotionBlur()
        super().__init__(perturbation)


class DefocusBlurStrategy(PerturbationStrategy):
    """Defocus blur perturbation strategy."""

    def __init__(self) -> None:
        """Initialize defocus blur strategy."""
        perturbation = DefocusBlur()
        super().__init__(perturbation)


class BoxBlurStrategy(PerturbationStrategy):
    """Box blur perturbation strategy."""

    def __init__(self) -> None:
        """Initialize box blur strategy."""
        perturbation = BoxBlur()
        super().__init__(perturbation)


# ============================================================================
# Lighting Perturbation Strategies
# ============================================================================


class GammaStrategy(PerturbationStrategy):
    """Gamma correction perturbation strategy."""

    def __init__(self) -> None:
        """Initialize gamma strategy."""
        perturbation = Gamma()
        super().__init__(perturbation)


class LowLightStrategy(PerturbationStrategy):
    """Low light perturbation strategy."""

    def __init__(self) -> None:
        """Initialize low light strategy."""
        perturbation = LowLight()
        super().__init__(perturbation)


class SaturationStrategy(PerturbationStrategy):
    """Saturation perturbation strategy."""

    def __init__(self) -> None:
        """Initialize saturation strategy."""
        perturbation = Saturation()
        super().__init__(perturbation)


# ============================================================================
# Spatial Perturbation Strategies
# ============================================================================


class RotationStrategy(PerturbationStrategy):
    """Rotation perturbation strategy."""

    def __init__(self) -> None:
        """Initialize rotation strategy."""
        perturbation = Rotation()
        super().__init__(perturbation)


class ScaleStrategy(PerturbationStrategy):
    """Scale perturbation strategy."""

    def __init__(self) -> None:
        """Initialize scale strategy."""
        perturbation = Scale()
        super().__init__(perturbation)


class TranslationStrategy(PerturbationStrategy):
    """Translation perturbation strategy."""

    def __init__(self) -> None:
        """Initialize translation strategy."""
        perturbation = Translation()
        super().__init__(perturbation)


class ShearStrategy(PerturbationStrategy):
    """Shear perturbation strategy."""

    def __init__(self) -> None:
        """Initialize shear strategy."""
        perturbation = Shear()
        super().__init__(perturbation)


class ElasticDeformStrategy(PerturbationStrategy):
    """Elastic deformation perturbation strategy."""

    def __init__(self) -> None:
        """Initialize elastic deformation strategy."""
        perturbation = ElasticDeform()
        super().__init__(perturbation)


# ============================================================================
# Composition Strategies
# ============================================================================


class ComposeStrategy(PerturbationStrategy):
    """Composition of multiple perturbations applied sequentially."""

    def __init__(self, perturbations: list) -> None:
        """Initialize composition strategy."""
        perturbation = Compose(perturbations)
        super().__init__(perturbation)


class RandomChoiceStrategy(PerturbationStrategy):
    """Random selection among multiple perturbations."""

    def __init__(self, perturbations: list) -> None:
        """Initialize random choice strategy."""
        perturbation = RandomChoice(perturbations)
        super().__init__(perturbation)


class BlendStrategy(PerturbationStrategy):
    """Blend of two perturbations."""

    def __init__(self, perturb1: Perturbation, perturb2: Perturbation) -> None:
        """Initialize blend strategy."""
        perturbation = Blend(perturb1, perturb2)
        super().__init__(perturbation)


class LowLightBlurStrategy(PerturbationStrategy):
    """Combined low light and blur perturbation."""

    def __init__(self) -> None:
        """Initialize low light blur strategy."""
        perturbation = LowLightBlur()
        super().__init__(perturbation)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Strategy",
    # Adversarial strategies
    "FGSMStrategy",
    "PGDStrategy",
    "BIMStrategy",
    "APGDStrategy",
    "SquareAttackStrategy",
    # Image perturbation strategies
    "GaussianNoiseStrategy",
    "BrightnessStrategy",
    "ContrastStrategy",
    "RotateStrategy",
    # Noise perturbation strategies
    "SaltPepperNoiseStrategy",
    "UniformNoiseStrategy",
    "SpeckleNoiseStrategy",
    # Blur perturbation strategies
    "GaussianBlurStrategy",
    "MotionBlurStrategy",
    "DefocusBlurStrategy",
    "BoxBlurStrategy",
    # Lighting perturbation strategies
    "GammaStrategy",
    "LowLightStrategy",
    "SaturationStrategy",
    # Spatial perturbation strategies
    "RotationStrategy",
    "ScaleStrategy",
    "TranslationStrategy",
    "ShearStrategy",
    "ElasticDeformStrategy",
    # Composition strategies
    "ComposeStrategy",
    "RandomChoiceStrategy",
    "BlendStrategy",
    "LowLightBlurStrategy",
]
