"""
This module defines the base Strategy class for all perturbation methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence


class Strategy:
    """
    Base class for all perturbation strategies in VisProbe.

    A strategy defines how to modify an input to test a model's robustness.
    """

    @classmethod
    def resolve(cls, perturb_spec: Any, *, level: Optional[float] = None) -> "Strategy":
        """
        Resolves a perturbation specification into a valid Strategy instance.

        Accepted forms:
        - Strategy instance (or any object with a `generate(imgs, model)` method)
        - Dict spec, e.g., {"type": "gaussian_noise", "std": 0.1}
        - Callable fn(level, imgs, model) -> imgs for ad-hoc perturbations
        - Sequence of any of the above → compose sequentially
        """
        # 0) Sequence → compose
        if isinstance(perturb_spec, (list, tuple)):
            strategies: List[Strategy] = [cls.resolve(s, level=level) for s in perturb_spec]
            return CompositeStrategy(strategies)
        # 1) Already a Strategy-like object
        if isinstance(perturb_spec, cls) or callable(getattr(perturb_spec, "generate", None)):
            return perturb_spec

        # 2) Dict-based specification
        if isinstance(perturb_spec, dict):
            if "type" not in perturb_spec:
                raise ValueError("Dict spec must include a 'type' field.")
            spec_type = perturb_spec["type"]
            params: Dict[str, Any] = {k: v for k, v in perturb_spec.items() if k != "type"}

            # Lazy import to avoid circular dependencies
            if spec_type in {"gaussian_noise", "brightness", "rotate"}:
                from .image import BrightnessStrategy, GaussianNoiseStrategy, RotateStrategy

                mapping: Dict[str, Callable[..., Strategy]] = {
                    "gaussian_noise": GaussianNoiseStrategy,
                    "brightness": BrightnessStrategy,
                    "rotate": RotateStrategy,
                }
                return mapping[spec_type](**params)
            if spec_type in {"fgsm", "pgd", "bim", "apgd", "square"}:
                from .adversarial import (
                    APGDStrategy,
                    BIMStrategy,
                    FGSMStrategy,
                    PGDStrategy,
                    SquareAttackStrategy,
                )

                mapping: Dict[str, Callable[..., Strategy]] = {
                    "fgsm": FGSMStrategy,
                    "pgd": PGDStrategy,
                    "bim": BIMStrategy,
                    "apgd": APGDStrategy,
                    "square": SquareAttackStrategy,
                }
                return mapping[spec_type](**params)
            raise ValueError(f"Unknown strategy type in dict spec: {spec_type}")

        # 3) Ad-hoc callable: fn(level, imgs, model) -> imgs
        if callable(perturb_spec):
            return _CallableStrategy(perturb_spec, level=level)

        raise ValueError(f"Unknown perturbation specification: {perturb_spec}.")

    def generate(self, imgs: Any, model: Any, level: Optional[float] = None) -> Any:
        """
        Generates a perturbed version of the input images.
        """
        raise NotImplementedError

    def apply(self, imgs: Any, model: Any, level: Optional[float] = None) -> Any:
        """
        Alias for generate() to maintain backward compatibility.

        Args:
            imgs: Input images to perturb
            model: The model being tested
            level: Optional perturbation level

        Returns:
            Perturbed images
        """
        return self.generate(imgs, model, level)

    def query_cost(self) -> int:
        """
        Returns the number of additional model queries used by the strategy.
        """
        return 0


class _CallableStrategy(Strategy):
    """Adapter for user-provided functions: fn(level, imgs, model) -> imgs."""

    def __init__(
        self, fn: Callable[[Optional[float], Any, Any], Any], *, level: Optional[float] = None
    ):
        self._fn = fn
        self.level = level

    def generate(self, imgs: Any, model: Any) -> Any:
        return self._fn(self.level, imgs, model)

    def query_cost(self) -> int:
        return 0


class CompositeStrategy(Strategy):
    """Composes multiple strategies sequentially: s_n(...s_2(s_1(x)))."""

    def __init__(self, strategies: Sequence[Strategy]):
        self.strategies: List[Strategy] = list(strategies)

    def generate(self, imgs: Any, model: Any, level: Optional[float] = None) -> Any:
        out = imgs
        for strat in self.strategies:
            # Some strategies accept level in generate; pass when provided
            try:
                out = strat.generate(out, model=model, level=level)
            except TypeError:
                out = strat.generate(out, model=model)
        return out

    def query_cost(self) -> int:
        # Sum of inner strategy query costs (best-effort; actual counted by QueryCounter)
        total = 0
        for strat in self.strategies:
            try:
                total += int(strat.query_cost())
            except Exception:
                pass
        return total
