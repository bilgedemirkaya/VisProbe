"""
Adversarial attack strategies using the Adversarial Robustness Toolbox (ART).

Provides gradient-based and score-based attacks for robustness testing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from .base import Strategy

logger = logging.getLogger(__name__)

# Lazy import flag for ART
_ART_AVAILABLE = False
_ART_IMPORT_ERROR: Optional[str] = None

try:
    from art.attacks.evasion import (
        AutoProjectedGradientDescent,
        BasicIterativeMethod,
        FastGradientMethod,
        ProjectedGradientDescent,
        SquareAttack,
    )
    from art.estimators.classification import PyTorchClassifier

    _ART_AVAILABLE = True
except ImportError as e:
    _ART_IMPORT_ERROR = str(e)
    PyTorchClassifier = None


def _check_art_available() -> None:
    """Raise ImportError if ART is not installed."""
    if not _ART_AVAILABLE:
        raise ImportError(
            "Adversarial strategies require the Adversarial Robustness Toolbox. "
            f"Install with: pip install adversarial-robustness-toolbox\n"
            f"Original error: {_ART_IMPORT_ERROR}"
        )


def _create_art_classifier(
    model: nn.Module, imgs: torch.Tensor, loss_fn: Optional[nn.Module] = None
) -> "PyTorchClassifier":
    """
    Create a PyTorchClassifier for ART attacks.

    Args:
        model: PyTorch model (or wrapper with .model attribute)
        imgs: Sample images to infer input shape and num classes
        loss_fn: Loss function (defaults to CrossEntropyLoss)

    Returns:
        Configured PyTorchClassifier instance
    """
    _check_art_available()

    # Unwrap if needed
    inner_model = getattr(model, "model", model)

    # Get device from model parameters
    try:
        model_device = next(inner_model.parameters()).device
    except StopIteration:
        model_device = imgs.device

    # Infer model properties
    sample = imgs[0:1].to(model_device)
    with torch.no_grad():
        output = inner_model(sample)
        if isinstance(output, tuple):
            output = output[0]

    return PyTorchClassifier(
        model=inner_model,
        loss=loss_fn or nn.CrossEntropyLoss(),
        input_shape=tuple(imgs[0].shape),
        nb_classes=output.shape[1],
        clip_values=(0.0, 1.0),
    )


class _ARTStrategyBase(Strategy):
    """Base class for ART-based strategies with classifier caching."""

    def __init__(self):
        self._cached_classifier = None
        self._cache_key = None

    def _get_classifier(self, model: nn.Module, imgs: torch.Tensor) -> "PyTorchClassifier":
        """Get or create cached classifier."""
        key = (id(model), imgs.shape[1:], str(imgs.device))
        if self._cache_key != key:
            self._cached_classifier = _create_art_classifier(model, imgs)
            self._cache_key = key
        return self._cached_classifier

    def _run_attack(self, attack, imgs: torch.Tensor) -> torch.Tensor:
        """Run ART attack and convert back to original device."""
        device = imgs.device
        adv_np = attack.generate(x=imgs.detach().cpu().numpy())
        return torch.from_numpy(adv_np).to(device)


class FGSMStrategy(_ARTStrategyBase):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Single-step attack that perturbs inputs in the direction of the loss gradient.

    Args:
        eps: Maximum perturbation (L∞ norm), default 2/255
        targeted: If True, minimize loss for target class
        art_attack_kwargs: Additional ART FastGradientMethod arguments
    """

    def __init__(
        self, eps: float = 2 / 255, targeted: bool = False, art_attack_kwargs: dict = None
    ):
        super().__init__()
        _check_art_available()
        self.eps = eps
        self.targeted = targeted
        self.art_attack_kwargs = art_attack_kwargs or {}

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        estimator = self._get_classifier(model, imgs)
        attack = FastGradientMethod(
            estimator=estimator,
            eps=level if level is not None else self.eps,
            targeted=self.targeted,
            **self.art_attack_kwargs,
        )
        return self._run_attack(attack, imgs)

    def query_cost(self) -> int:
        return 1

    def __str__(self) -> str:
        return f"FGSMStrategy(eps={self.eps:.4f})"

    def __repr__(self) -> str:
        return f"FGSMStrategy(eps={self.eps}, targeted={self.targeted})"


class PGDStrategy(_ARTStrategyBase):
    """
    Projected Gradient Descent (PGD) attack.

    Iterative attack that takes multiple smaller steps with projection.
    Stronger than FGSM but requires more queries.

    Args:
        eps: Maximum perturbation (L∞ norm)
        eps_step: Step size per iteration (default: eps/10)
        max_iter: Maximum iterations (default: 100)
    """

    def __init__(
        self, eps: float, eps_step: Optional[float] = None, max_iter: int = 100, **kwargs: Any
    ):
        super().__init__()
        _check_art_available()
        self.eps = eps
        self.eps_step = eps_step if eps_step is not None else eps / 10
        self.max_iter = max_iter
        self.art_attack_kwargs = kwargs

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        estimator = self._get_classifier(model, imgs)
        attack = ProjectedGradientDescent(
            estimator=estimator,
            eps=level if level is not None else self.eps,
            eps_step=self.eps_step,
            max_iter=self.max_iter,
            **self.art_attack_kwargs,
        )
        return self._run_attack(attack, imgs)

    def query_cost(self) -> int:
        return self.max_iter

    def __str__(self) -> str:
        return f"PGDStrategy(eps={self.eps:.4f}, iter={self.max_iter})"

    def __repr__(self) -> str:
        return f"PGDStrategy(eps={self.eps}, eps_step={self.eps_step}, max_iter={self.max_iter})"


class BIMStrategy(_ARTStrategyBase):
    """
    Basic Iterative Method (BIM) attack.

    Also known as I-FGSM. Iteratively applies FGSM with smaller steps.

    Args:
        eps: Maximum perturbation (L∞ norm)
        eps_step: Step size per iteration (default: eps/max_iter)
        max_iter: Maximum iterations (default: 10)
    """

    def __init__(
        self, eps: float, eps_step: Optional[float] = None, max_iter: int = 10, **kwargs: Any
    ):
        super().__init__()
        _check_art_available()
        self.eps = eps
        self.eps_step = eps_step if eps_step is not None else eps / max(1, max_iter)
        self.max_iter = max_iter
        self.art_attack_kwargs = kwargs

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        estimator = self._get_classifier(model, imgs)
        attack = BasicIterativeMethod(
            estimator=estimator,
            eps=level if level is not None else self.eps,
            eps_step=self.eps_step,
            max_iter=self.max_iter,
            **self.art_attack_kwargs,
        )
        return self._run_attack(attack, imgs)

    def query_cost(self) -> int:
        return self.max_iter

    def __str__(self) -> str:
        return f"BIMStrategy(eps={self.eps:.4f}, iter={self.max_iter})"

    def __repr__(self) -> str:
        return f"BIMStrategy(eps={self.eps}, eps_step={self.eps_step}, max_iter={self.max_iter})"


class APGDStrategy(_ARTStrategyBase):
    """
    Auto-PGD (APGD) attack.

    Adaptive step-size PGD with automatic hyperparameter tuning.
    Supports CE and DLR loss variants via kwargs.

    Args:
        eps: Maximum perturbation (L∞ norm)
        max_iter: Maximum iterations (default: 100)
    """

    def __init__(self, eps: float, max_iter: int = 100, **kwargs: Any):
        super().__init__()
        _check_art_available()
        self.eps = eps
        self.max_iter = max_iter
        self.art_attack_kwargs = kwargs

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        estimator = self._get_classifier(model, imgs)
        attack = AutoProjectedGradientDescent(
            estimator=estimator,
            eps=level if level is not None else self.eps,
            max_iter=self.max_iter,
            **self.art_attack_kwargs,
        )
        return self._run_attack(attack, imgs)

    def query_cost(self) -> int:
        return self.max_iter

    def __str__(self) -> str:
        return f"APGDStrategy(eps={self.eps:.4f}, iter={self.max_iter})"

    def __repr__(self) -> str:
        return f"APGDStrategy(eps={self.eps}, max_iter={self.max_iter})"


class SquareAttackStrategy(_ARTStrategyBase):
    """
    Square Attack (score-based black-box attack).

    Query-efficient black-box attack using random square-shaped perturbations.
    Does not require gradients.

    Args:
        eps: Maximum perturbation (L∞ norm)
        max_iter: Maximum queries (default: 5000)
    """

    def __init__(self, eps: float, max_iter: int = 5000, **kwargs: Any):
        super().__init__()
        _check_art_available()
        self.eps = eps
        self.max_iter = max_iter
        self.art_attack_kwargs = kwargs

    def generate(
        self, imgs: torch.Tensor, model: nn.Module, level: Optional[float] = None
    ) -> torch.Tensor:
        estimator = self._get_classifier(model, imgs)
        attack = SquareAttack(
            estimator=estimator,
            eps=level if level is not None else self.eps,
            max_iter=self.max_iter,
            **self.art_attack_kwargs,
        )
        return self._run_attack(attack, imgs)

    def query_cost(self) -> int:
        return self.max_iter

    def __str__(self) -> str:
        return f"SquareAttackStrategy(eps={self.eps:.4f}, iter={self.max_iter})"

    def __repr__(self) -> str:
        return f"SquareAttackStrategy(eps={self.eps}, max_iter={self.max_iter})"
