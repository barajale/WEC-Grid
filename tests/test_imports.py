"""
Basic import tests for WEC-GRID core modules.
"""

import pytest


def test_database_import():
    """Test database module can be imported."""
    from wecgrid.util.database import WECGridDB
    assert WECGridDB is not None


def test_time_manager_import():
    """Test time manager can be imported."""
    from wecgrid.util.time import WECGridTimeManager
    assert WECGridTimeManager is not None


def test_plot_import():
    """Test plotting module can be imported."""
    from wecgrid.plot.plot import WECGridPlot
    assert WECGridPlot is not None


def test_grid_state_import():
    """Test GridState can be imported."""
    from wecgrid.modelers.power_system.base import GridState
    assert GridState is not None


def test_engine_import():
    """Test Engine can be imported (may skip if external deps missing)."""
    try:
        from wecgrid.engine import Engine
        assert Engine is not None
    except ImportError as e:
        # Skip if external dependencies are missing
        if any(x in str(e).lower() for x in ["matlab", "psspy", "psse"]):
            pytest.skip(f"External dependency not available: {e}")
        else:
            raise
