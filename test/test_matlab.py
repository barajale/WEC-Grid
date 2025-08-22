"""Verify MATLAB engine availability."""

import pytest


@pytest.mark.matlab
def test_matlab():
    """Test MATLAB modules can be imported."""
    engine = pytest.importorskip("matlab.engine")
    assert engine is not None

