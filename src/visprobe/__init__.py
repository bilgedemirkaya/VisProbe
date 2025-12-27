"""
VisProbe: Find robustness failures in your vision models in 5 minutes.

Quick Start:
    >>> from visprobe import quick_check
    >>> report = quick_check(model, data, preset="natural")
    >>> report.show()

For single-strategy threshold finding:
    >>> from visprobe import search
    >>> from visprobe.strategies.image import GaussianNoiseStrategy
    >>> report = search(model, data, strategy=lambda l: GaussianNoiseStrategy(std_dev=l))
"""

__version__ = "0.2.0"

# Primary API
from .quick import quick_check, compare_threat_models
from .search import search
from .report import Report

# Modules
from . import presets, properties, strategies

__all__ = [
    # Primary API
    "quick_check",          # Multi-strategy preset testing
    "search",               # Single-strategy threshold finding
    "compare_threat_models",  # Compare all threat models
    "Report",               # Report class
    # Modules
    "properties",
    "strategies",
    "presets",
]
