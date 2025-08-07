from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime

@dataclass
class WECDevice:
    """
    Represents a single Wave Energy Converter (WEC) device.
    """
    name: str
    parameters: Dict[str, Any]
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    Pmax: float = 9999.0
    Pmin: float = -9999.0
    Qmax: float = 9999.0
    Qmin: float = -9999.0
    MBASE: float = 0.1  # default base = 100 kW
    bus_location: Optional[int] = None
    model: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def attach_time_index(self, start_time: datetime, freq: str = "5T") -> None:
        """
        Attach datetime index to the dataframe.
        """
        if not self.dataframe.empty:
            snapshots = pd.date_range(
                start=start_time,
                periods=self.dataframe.shape[0],
                freq=freq
            )
            self.dataframe["snapshots"] = snapshots
            self.dataframe.set_index("snapshots", inplace=True)

    def load_from_database(self, db_query_func, wec_id: int) -> bool:
        """
        Pulls WEC data from the database using a query function.
        """
        table_name = f"WEC_output_{wec_id}"
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = db_query_func(check_query)

        if not check_result or check_result[0][0] != table_name:
            return False
    

        data_query = f"SELECT * FROM {table_name}"
        self.dataframe = db_query_func(data_query, return_type="df")
        return True

    def __repr__(self):
        return f"<Device name={self.name} bus={self.bus_location} model={self.model} rows={len(self.dataframe)}>"

