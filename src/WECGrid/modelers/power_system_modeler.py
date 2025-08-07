# power_system_modeler.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import pandas as pd
from .network_state import NetworkState


class PowerSystemModeler(ABC):
    def __init__(self, engine: Any):
        self.engine = engine
        self.state = NetworkState()

    @abstractmethod
    def init_api(self) -> bool:
        """Initialize backend API and load network into memory."""
        pass

    @abstractmethod
    def solve_powerflow(self) -> bool:
        #"""Run a power flow solution and update `self.state`."""
        """Run a power flow solution, not updating state"""
        pass

    @abstractmethod
    def add_wec(self, model: str, from_bus: int, to_bus: int) -> bool:
        """Inject a WEC into the grid model."""
        pass

    @abstractmethod
    def simulate(self, load_curve: bool = True, plot: bool = True) -> bool:
        """Run full time-series simulation."""
        pass

    @abstractmethod
    def take_snapshot(self, timestamp: pd.Timestamp) -> None:
        """Capture and store current network state into `self.state`."""
        pass

    @property
    def bus(self) -> Optional[pd.DataFrame]:
        return self.state.bus

    @property
    def bus_t(self) -> Dict[str, pd.DataFrame]:
        return self.state.bus_t