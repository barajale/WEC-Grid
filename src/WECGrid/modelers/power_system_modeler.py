"""
Abstract base class for power‐system modelers
"""

import abc
from typing import Any

class PowerSystemModeler(abc.ABC):
    def __init__(self, case_file: str, engine: Any):
        """
        Args:
            case_file: path to the RAW/PYPSA network file
            engine: reference back to the Engine orchestrator
        """
        self.case_file = case_file
        self.engine    = engine

    @abc.abstractmethod
    def init_api(self) -> None:
        """
        Initialize the external API (PSS®E or PyPSA) and load the network.
        """
        pass

    # you can add other abstract methods here,
    # e.g. solve_powerflow, get_bus_dataframe_t, get_generator_dataframe_t, etc.