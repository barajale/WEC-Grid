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
    MBASE: float = 0.1  # default base = 100 kW
    bus_location: Optional[int] = None
    model: Optional[str] = None
    sim_id: Optional[int] = None

    def __repr__(self) -> str:
        return f"""WECDevice:
    ├─ name: {self.name!r}
    ├─ model: {self.model!r}
    ├─ bus_location: {self.bus_location}
    ├─ sim_id: {self.sim_id}
    ├─ MBASE: {self.MBASE}
    └─ rows: {len(self.dataframe)}
    """