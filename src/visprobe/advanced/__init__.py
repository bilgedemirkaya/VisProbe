"""
Advanced decorator-based API for VisProbe (for power users).

⚠️  DEPRECATION NOTICE:
The decorator-based API is deprecated and will be removed in version 2.0.
Please use the simpler `quick_check()` API instead:

    from visprobe import quick_check
    report = quick_check(model, data, preset="standard")
    report.show()

See README.md for migration guide.

The decorator API remains available for power users who need fine-grained control.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "The decorator-based API (visprobe.advanced) is deprecated and will be "
    "removed in version 2.0. Please use the quick_check() API instead. "
    "See README.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

# Import decorator API from main package
from ..api import (
    ImageData,
    PerturbationInfo,
    Report,
    data_source,
    given,
    model,
    search,
)

__all__ = [
    "given",
    "model",
    "data_source",
    "search",
    "Report",
    "ImageData",
    "PerturbationInfo",
]
