"""Synthetic data generation utilities for testing and development.

This package supports optional "advanced" writers for DOCX/XLSX/PPTX generation.
Those features require additional third-party dependencies. Imports are guarded so
the base utilities remain usable in minimal environments.
"""

from .config import *  # noqa: F403
from .file_writers import FileWriter
from .formatters import DataFormatter
from .generators import PIIGenerator

try:
    from .advanced_file_writers import AdvancedFileWriter
    from .advanced_formatters import AdvancedDataFormatter
    from .dataset_generator import MultiFormatDatasetGenerator
except Exception:  # pragma: no cover
    AdvancedDataFormatter = None  # type: ignore
    AdvancedFileWriter = None  # type: ignore
    MultiFormatDatasetGenerator = None  # type: ignore

__all__ = [
    "TAIWAN_LOCATIONS",
    "STREET_NAMES",
    "SURNAMES",
    "GIVEN_NAMES",
    "HOSPITALS",
    "MEDICAL_SPECIALTIES",
    "PIIGenerator",
    "DataFormatter",
    "FileWriter",
    "AdvancedDataFormatter",
    "AdvancedFileWriter",
    "MultiFormatDatasetGenerator",
]
