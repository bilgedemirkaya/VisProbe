"""
Perturbations package for visfuzz.

Provides GPU-optimized image perturbations for robustness testing.

Categories:
- noise: Gaussian, salt-pepper, uniform, speckle noise
- blur: Gaussian, motion, defocus, box blur
- lighting: Brightness, contrast, gamma, low-light, saturation
- spatial: Rotation, scale, translation, shear, elastic deformation
- composition: Combine multiple perturbations
"""

from .base import BatchedPerturbation, Perturbation
from .blur import BoxBlur, DefocusBlur, GaussianBlur, MotionBlur
from .composition import Blend, Compose, LowLightBlur, RandomChoice
from .lighting import Brightness, Contrast, Gamma, LowLight, Saturation
from .noise import GaussianNoise, SaltPepperNoise, SpeckleNoise, UniformNoise
from .spatial import ElasticDeform, Rotation, Scale, Shear, Translation

__all__ = [
    # Base
    "Perturbation",
    "BatchedPerturbation",
    # Noise
    "GaussianNoise",
    "SaltPepperNoise",
    "UniformNoise",
    "SpeckleNoise",
    # Blur
    "GaussianBlur",
    "MotionBlur",
    "DefocusBlur",
    "BoxBlur",
    # Lighting
    "Brightness",
    "Contrast",
    "Gamma",
    "LowLight",
    "Saturation",
    # Spatial
    "Rotation",
    "Scale",
    "Translation",
    "Shear",
    "ElasticDeform",
    # Composition
    "Compose",
    "RandomChoice",
    "Blend",
    "LowLightBlur",
]


# Registry of all perturbations by name
PERTURBATION_REGISTRY = {
    # Noise
    "gaussian_noise": GaussianNoise,
    "salt_pepper_noise": SaltPepperNoise,
    "uniform_noise": UniformNoise,
    "speckle_noise": SpeckleNoise,
    # Blur
    "gaussian_blur": GaussianBlur,
    "motion_blur": MotionBlur,
    "defocus_blur": DefocusBlur,
    "box_blur": BoxBlur,
    # Lighting
    "brightness": Brightness,
    "contrast": Contrast,
    "gamma": Gamma,
    "low_light": LowLight,
    "saturation": Saturation,
    # Spatial
    "rotation": Rotation,
    "scale": Scale,
    "translation": Translation,
    "shear": Shear,
    "elastic_deform": ElasticDeform,
    # Composition
    "low_light_blur": LowLightBlur,
}


def get_perturbation(name: str, **kwargs) -> Perturbation:
    """
    Get a perturbation by name.

    Args:
        name: Perturbation name (e.g., 'gaussian_noise', 'rotation')
        **kwargs: Arguments to pass to the perturbation constructor

    Returns:
        Perturbation instance

    Raises:
        ValueError: If perturbation name is unknown
    """
    if name not in PERTURBATION_REGISTRY:
        available = ", ".join(sorted(PERTURBATION_REGISTRY.keys()))
        raise ValueError(f"Unknown perturbation: {name}. Available: {available}")

    return PERTURBATION_REGISTRY[name](**kwargs)
