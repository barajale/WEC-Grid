"""
Simulation runner for a WEC farm
"""

# typing
from typing import Dict, Any
# bring Farm into scope
from .farm import Farm

class SimRunner:
    def __init__(self, farm: Farm, settings: Dict[str, Any]):
        """
        Args:
            farm:     instance of wecgrid.wec.Farm
            settings: dict of simulation parameters
        """
        self.farm     = farm
        self.settings = settings

    # … rest of your methods here …