"""Load scripts/01_prepare_data.py as an importable module.

The script's filename starts with a digit (for run-order clarity:
00_, 01_, 02_...), so it isn't a valid target for a plain `import`
statement — load it by file path instead.
"""
import importlib.util
import os

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_module(filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.rstrip(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def prepare_data():
    return _load_module("01_prepare_data.py")


@pytest.fixture(scope="session")
def build_hourly_graphs_module():
    return _load_module("05_build_hourly_graphs.py")


@pytest.fixture(scope="session")
def graph_features_module():
    return _load_module("06_compute_graph_features.py")


@pytest.fixture(scope="session")
def enrich_module():
    return _load_module("07_enrich_with_graph_features.py")


@pytest.fixture(scope="session")
def classifier_graph_module():
    return _load_module("08_classifier_with_graph_features.py")


@pytest.fixture(scope="session")
def shap_analysis_module():
    return _load_module("09_shap_analysis.py")


@pytest.fixture(scope="session")
def chain_module():
    return _load_module("10_extract_lateral_movement_chain.py")


@pytest.fixture(scope="session")
def mttd_module():
    return _load_module("11_mttd_analysis.py")


@pytest.fixture(scope="session")
def perflow_novelty_module():
    return _load_module("14_perflow_novelty_eval.py")
