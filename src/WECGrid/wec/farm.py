"""
Wave Energy Converter Farm class
"""

from typing import List, Dict, Any
import pandas as pd
from .device import Device
from .wec_sim_runner import WECSimRunner
from ..database.wecgrid_db import dbQuery


class Farm:
    def __init__(self, engine, sim_id: int, model: str, bus_location: int, size: int = 1, config: Dict[str, Any] = {}):
        """
        Represents a collection of WEC devices sharing the same model and connection bus.

        Args:
            engine: Reference to the main Engine instance.
            sim_id: Simulation identifier (shared across devices).
            model: WEC model name (e.g., "RM3").
            bus_location: Bus number in the grid where devices are connected.
            size: Number of identical devices in the farm.
            config: Configuration dictionary passed to WEC-SIM.
        """
        self.engine = engine
        self.sim_id = sim_id
        self.model = model
        self.bus_location = bus_location
        self.size = size
        self.config = config
        self.devices: List[Device] = []

        self.db = engine.db  # assumes engine exposes `wecgrid_db` instance
        self.runner = WECSimRunner()

        # Initialize devices and run sim if needed
        self._prepare_farm()

    def _prepare_farm(self):
        """
        Attempts to pull WEC data from the database. If missing, runs WEC-SIM.
        Then instantiates all device objects.
        """
        table_name = f"WEC_output_{self.sim_id}"
        exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        result = dbQuery(exists_query)

        if not result:
            print(f"[Farm] No WEC data for sim_id={self.sim_id} found in database. Running WEC-SIM...")
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
        query = f"SELECT * FROM {table_name}"
        df = dbQuery(query, return_type="df")
        if df is None or df.empty:
            raise RuntimeError(f"[Farm] Failed to load WEC data for sim_id={self.sim_id}")

        for i in range(self.size):
            name = f"{self.model}_{self.sim_id}_{i}"
            device = Device(name=name, parameters={"bus": self.bus_location}, dataframe=df.copy())
            self.devices.append(device)