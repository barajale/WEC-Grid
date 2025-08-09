from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import pandas as pd
from datetime import datetime

@dataclass
class WECDevice:
    """
    Represents a single Wave Energy Converter (WEC) device.
    """
    name: str
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    dataframe_full: pd.DataFrame = field(default_factory=pd.DataFrame)
    base: Optional[float] = None  # in MW
    bus_location: Optional[int] = None
    model: Optional[str] = None
    sim_id: Optional[int] = None

    def __repr__(self) -> str:
        return f"""WECDevice:
    ├─ name: {self.name!r}
    ├─ model: {self.model!r}
    ├─ bus_location: {self.bus_location}
    ├─ sim_id: {self.sim_id}
    ├─ base: {"{} MW".format(self.base)}
    └─ rows: {len(self.dataframe)}
    """