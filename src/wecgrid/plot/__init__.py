"""
WEC-Grid plotting tools
"""

from .plot import WECGridPlot

# Keep old name for backward compatibility
WECGridPlotter = WECGridPlot

__all__ = [
    "WECGridPlot",
    "WECGridPlotter"  # Backward compatibility
]