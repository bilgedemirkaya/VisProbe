"""
VisProbe: Find robustness failures in your vision models in 5 minutes.

Quick Start:
    >>> from visprobe import quick_check
    >>> report = quick_check(model, data, preset="standard")
    >>> report.show()

For advanced usage, see visprobe.advanced
"""

__version__ = "0.1.0"

# Primary API - simple and user-friendly
from .quick import quick_check
from .api import Report

# Modules
from . import auto_init, presets, properties, strategies

__all__ = [
    # Primary API (use this!)
    "quick_check",
    "Report",
    # Modules
    "properties",
    "strategies",
    "presets",
    "auto_init",
]

# Backward compatibility: Keep decorators available but deprecated
# Users should migrate to quick_check()
def __getattr__(name):
    """Lazy import with deprecation warning for old decorator API."""
    import warnings

    deprecated_names = ["given", "model", "data_source", "search", "ImageData", "PerturbationInfo"]

    if name in deprecated_names:
        warnings.warn(
            f"Importing '{name}' from visprobe is deprecated. "
            f"Please use quick_check() instead, or import from visprobe.advanced "
            f"if you need the decorator API. See README.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .api import (
            ImageData,
            PerturbationInfo,
            data_source,
            given,
            model,
            search,
        )
        return locals()[name]

    raise AttributeError(f"module 'visprobe' has no attribute '{name}'")
