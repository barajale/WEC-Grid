"""
Wave Energy Converter Farm class
"""

from typing import List, Dict, Any
import pandas as pd
from .wecdevice import WECDevice
from .wecsim_runner import WECSimRunner


class WECFarm:
    def __init__(self, farm_name: str, database, time: Any, sim_id: int, model: str, bus_location: int, connecting_bus: int = 1, size: int = 1):
        """
        Represents a collection of WEC devices sharing the same model and connection bus.

        Args:
            engine: Reference to the main Engine instance.
            sim_id: Simulation identifier (shared across devices).
            model: WEC model name (e.g., "RM3").
            bus_location: Bus number in the grid where devices are connected.
            size: Number of identical devices in the farm.
        """
        
        self.farm_name: str = farm_name
        self.database = database # TODO make this a WECGridDB data type
        self.time = time # todo might need to update time to be SimulationTime type 
        self.sim_id: int = sim_id
        self.model: str = model
        self.bus_location: int = bus_location
        self.connecting_bus: int = connecting_bus # todo this should default to swing bus
        self.gen_id: str = f"W{size}"
        self.size: int = size
        self.config: Dict = None
        self.wec_devices: List[WECDevice] = []
        self.MBASE: float = 0.1  # default base = 100 kW

        self._prepare_farm()

    def __repr__(self) -> str:
        return f"""WECFarm:
        ├─ name: {self.farm_name!r}
        ├─ size: {len(self.wec_devices)}
        ├─ model: {self.model!r}
        ├─ bus_location: {self.bus_location}
        ├─ connecting_bus: {self.connecting_bus}
        └─ sim_id: {self.sim_id}
        
    """
    def _prepare_farm(self):
        """
        Attempts to pull WEC data from the database.
        """
        table_name = f"WEC_output_{self.sim_id}"
        exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        result = self.database.query(exists_query)

        if not result:
            raise RuntimeError(f"[Farm] No WEC data for sim_id={self.sim_id} found in database. Run WEC-SIM first.")
            # TODO: Provide clearer guidance on running WEC-SIM

        # Load data once and distribute to all devices
        df = self.database.query(f"SELECT * FROM {table_name}", return_type="df")
        if df is None or df.empty:
            raise RuntimeError(f"[Farm] Failed to load WEC data for sim_id={self.sim_id}")

        # Apply time index at 5 min resolution using start time
        df["snapshots"] = pd.date_range(start=self.time.start_time, periods=df.shape[0], freq="5T")
        df.set_index("snapshots", inplace=True) 

        for i in range(self.size):
            name = f"{self.model}_{self.sim_id}_{i}"
            device = WECDevice(
                name=name,
                dataframe=df.copy(),
                bus_location=self.bus_location,
                model=self.model,
                sim_id=self.sim_id
            )
            self.wec_devices.append(device)
            
        
        
    def power_at_snapshot(self, timestamp: pd.Timestamp) -> float:
        """
        Returns the total power output of the WEC farm at a given snapshot timestamp.

        Args:
            timestamp: The snapshot time to query (must exist in the DataFrame index).
            power_col: Name of the power column in the device DataFrame (default: "P_WEC").

        Returns:
            Total power output across all devices at the given timestamp [in MW].
        """
        total_power = 0.0
        for device in self.wec_devices:
            if (
                device.dataframe is not None 
                and not device.dataframe.empty 
                and timestamp in device.dataframe.index
            ):
                power = device.dataframe.at[timestamp, "pg"]
                total_power += power
            else:
                print(f"[WARNING] Missing data for {device.name} at {timestamp}")
        return total_power