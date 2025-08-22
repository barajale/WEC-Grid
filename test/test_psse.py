"""Tests for PSS®E integration."""

import pytest


@pytest.mark.psse
def test_psse():
    """Test PSS®E modules can be imported."""
    pssepath = pytest.importorskip("pssepath")
    pssepath.add_pssepath()
    psspy = pytest.importorskip("psspy")
    psse35 = pytest.importorskip("psse35")
    assert psspy is not None
    assert psse35 is not None

