# power_system_modeler.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime        # <- use datetime to match PSSEModeler
import pandas as pd
from .grid_state import GridState
from ..wec.wecfarm import WECFarm


class PowerSystemModeler(ABC):
    def __init__(self, engine: Any):
        self.engine = engine
        self.grid = GridState()
        self.sbase: Optional[float] = None
        

    @abstractmethod
    def init_api(self) -> bool:
        """Initialize backend API and load grid into memory."""
        pass

    @abstractmethod
    def solve_powerflow(self) -> bool:
        """Run a power flow solution (return True on success)."""
        pass

    @abstractmethod
    def add_wec_farm(self, farm: WECFarm) -> bool:
        """Inject a WEC farm into the grid model."""
        pass

    @abstractmethod
    def simulate(self,
                 load_curve: Optional[pd.DataFrame] = None) -> bool:
        """Run full time-series simulation."""
        pass

    @abstractmethod
    def take_snapshot(self, timestamp: datetime) -> None:
        """Capture and store current grid state into `self.grid`."""
        pass

    # Convenience accessors
    @property
    def bus(self) -> Optional[pd.DataFrame]:
        return self.grid.bus

    @property
    def gen(self) -> Optional[pd.DataFrame]:
        return self.grid.gen

    @property
    def load(self) -> Optional[pd.DataFrame]:
        return self.grid.load

    @property
    def line(self) -> Optional[pd.DataFrame]:
        return self.grid.line

    @property
    def bus_t(self) -> Dict[str, pd.DataFrame]:
        return self.grid.bus_t

    @property
    def gen_t(self) -> Dict[str, pd.DataFrame]:
        return self.grid.gen_t

    @property
    def load_t(self) -> Dict[str, pd.DataFrame]:
        return self.grid.load_t

    @property
    def line_t(self) -> Dict[str, pd.DataFrame]:
        return self.grid.line_t