# SPDX-License-Identifier: MIT
"""Shared test fixtures and path helpers."""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(REPO_ROOT / 'data')
EXAMPLES_DIR = REPO_ROOT / 'examples'


@pytest.fixture(scope='session')
def data_dir():
    """Absolute path to the SN data directory."""
    return DATA_DIR


@pytest.fixture(scope='session')
def example_plugin_path():
    """Absolute path to examples/test_external.py."""
    return str(EXAMPLES_DIR / 'test_external.py')
