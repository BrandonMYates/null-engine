# SPDX-License-Identifier: MIT
"""Null Engine: empirical null calibration for dark-energy Δχ² tests.

Public API:
  MODELS                       Registry of built-in dark-energy parameterizations
  init_worker, run_one_mock    Multiprocessing pool helpers
  main                         CLI entry point (`null-engine ...`)
  normalize_model_spec         User-facing model dict normalizer
  load_model_file              Load an external plugin .py
  register_external_model      Register a plugin into MODELS
  validate_model               Small-batch sanity validation harness
  print_validation_report      Pretty-print a validation report
"""

from null_engine.models import MODELS
from null_engine.engine import init_worker, run_one_mock, main
from null_engine.model_spec import normalize_model_spec
from null_engine.plugin_loader import load_model_file, register_external_model
from null_engine.validator import validate_model, print_validation_report

__version__ = "1.0.0"

__all__ = [
    "MODELS",
    "init_worker",
    "run_one_mock",
    "main",
    "normalize_model_spec",
    "load_model_file",
    "register_external_model",
    "validate_model",
    "print_validation_report",
    "__version__",
]
