"""
VisProbe: A library for adversarial robustness testing and visualization.
"""

__version__ = "0.1.0"

# Auto-initialization is available as a separate import
# Users can do: import visprobe.auto_init
from . import auto_init, properties, strategies
from .api import (
    ImageData,
    PerturbationInfo,
    Report,
    data_source,
    given,
    model,
    search,
)

__all__ = [
    # Decorators
    "given",
    "model",
    "data_source",
    "search",
    # Core classes
    "Report",
    "ImageData",
    "PerturbationInfo",
    # Modules
    "properties",
    "strategies",
    "auto_init",
]
