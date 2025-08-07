"""
Wave Energy Converter Farm class
"""

from typing import List, Dict, Any
import pandas as pd
from .wecdevice import WECDevice
from .wecsim_runner import WECSimRunner


class WECFarm:
    def __init__(self, database, paths, sim_id: int, model: str, bus_location: int, size: int = 1):
        """
        Represents a collection of WEC devices sharing the same model and connection bus.

        Args:
            engine: Reference to the main Engine instance.
            sim_id: Simulation identifier (shared across devices).
            model: WEC model name (e.g., "RM3").
            bus_location: Bus number in the grid where devices are connected.
            size: Number of identical devices in the farm.
        """
        self.database = database # TODO make this a WECGridDB data type
        
        self.sim_id: int = sim_id
        self.model: str = model
        self.bus_location: int = bus_location
        self.size: int = size
        self.config: Dict = None
        self.devices: List[WECDevice] = []

        self._prepare_farm()


    def _prepare_farm(self):
        """
        Attempts to pull WEC data from the database.
        """
        table_name = f"WEC_output_{self.sim_id}"
        exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        result = self.database.query(exists_query)

        if not result:
            print(f"[Farm] No WEC data for sim_id={self.sim_id} found in database.")
            success = self.runner(
                sim_id=self.sim_id,
                model=self.model,
                config=self.config,
            )
            if not success:
                raise RuntimeError(f"[Farm] WEC-SIM failed for sim_id={self.sim_id}")

        else:
            print(f"[Farm] Found WEC data for sim_id={self.sim_id} in database.")

        # Load data once and distribute to all devices
        df = self.database.query(f"SELECT * FROM {table_name}")
        if df is None or df.empty:
            raise RuntimeError(f"[Farm] Failed to load WEC data for sim_id={self.sim_id}")

        for i in range(self.size):
            name = f"{self.model}_{self.sim_id}_{i}"
            device = Device(name=name, parameters={"bus": self.bus_location}, dataframe=df.copy())
            self.devices.append(device)
            
    