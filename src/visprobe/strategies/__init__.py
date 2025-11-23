"""
The `visprobe.strategies` module provides a collection of perturbation
strategies that can be used in `@given` and `@search` tests.
"""

from __future__ import annotations

from .adversarial import APGDStrategy, BIMStrategy, FGSMStrategy, PGDStrategy, SquareAttackStrategy
from .base import Strategy
from .image import BrightnessStrategy, ContrastStrategy, GaussianNoiseStrategy, RotateStrategy

__all__ = [
    "Strategy",
    "FGSMStrategy",
    "PGDStrategy",
    "BIMStrategy",
    "APGDStrategy",
    "SquareAttackStrategy",
    "GaussianNoiseStrategy",
    "BrightnessStrategy",
    "ContrastStrategy",
    "RotateStrategy",
]
