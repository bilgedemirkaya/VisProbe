"""
This module defines the base class and result type for all properties in VisProbe.
"""

from __future__ import annotations

from typing import Any


class Property:
    """
    Base class for all properties in VisProbe.

    A property is a function or class that asserts a specific behavior
    of a model, typically by comparing its output on an original and a
    perturbed input.
    """

    def __call__(self, original: Any, perturbed: Any) -> bool:
        """
        Executes the property check.

        Args:
            original: The output of the model on the original input.
            perturbed: The output of the model on the perturbed input.

        Returns:
            True if the property holds, False otherwise.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    @classmethod
    def evaluate(cls, original: Any, perturbed: Any, **init_kwargs) -> bool:
        """
        Convenience single-call API.

        Usage:
            - Paramless property: MyProp.evaluate(original, perturbed)
            - With parameters:    TopKStability.evaluate(original, perturbed, k=5)
        """
        instance = cls(**init_kwargs)  # type: ignore[arg-type]
        return instance(original, perturbed)
