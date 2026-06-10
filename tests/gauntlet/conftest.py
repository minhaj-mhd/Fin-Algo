import os
import pytest

@pytest.fixture(autouse=True)
def set_test_gauntlet_root(tmp_path):
    """
    Automatically sets GAUNTLET_ROOT environment variable to a temporary path
    for all gauntlet unit tests, ensuring no production files are touched.
    """
    old_root = os.environ.get("GAUNTLET_ROOT")
    os.environ["GAUNTLET_ROOT"] = str(tmp_path)
    yield
    if old_root is not None:
        os.environ["GAUNTLET_ROOT"] = old_root
    else:
        del os.environ["GAUNTLET_ROOT"]
