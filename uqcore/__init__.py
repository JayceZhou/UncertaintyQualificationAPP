"""SciUQ Studio numerical analysis package."""

from .classification import ClassificationResult, analyze_mc_probabilities
from .niw import NIWResult, analyze_niw_field

__all__ = [
    "ClassificationResult",
    "NIWResult",
    "analyze_mc_probabilities",
    "analyze_niw_field",
]

