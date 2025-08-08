# File: src/wecgrid/util/wecgrid_timemanager.py

from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

@dataclass
class WECGridTimeManager:
    """
    Manages simulation time: start time, end time, interval, and snapshots.
    """
    start_time: datetime = field(default_factory=lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    sim_length: int = 288
    freq: str = "5T"

    def __post_init__(self):
        self._update_sim_stop()

    def _update_sim_stop(self):
        self.sim_stop = self.snapshots[-1] if self.sim_length > 0 else self.start_time

    @property
    def snapshots(self) -> pd.DatetimeIndex:
        return pd.date_range(
            start=self.start_time,
            periods=self.sim_length,
            freq=self.freq,
        )

    def update(self, *, start_time: datetime = None, sim_length: int = None, freq: str = None):
        if start_time is not None:
            self.start_time = start_time
        if sim_length is not None:
            self.sim_length = sim_length
        if freq is not None:
            self.freq = freq
        self._update_sim_stop()

    def set_end_time(self, end_time: datetime):
        self.sim_length = len(pd.date_range(start=self.start_time, end=end_time, freq=self.freq))
        self.sim_stop = end_time

    def __repr__(self) -> str:
        return (
            f"WECGridTimeManager:\n"
            f"├─ start_time: {self.start_time}\n"
            f"├─ sim_stop:   {self.sim_stop}\n"
            f"├─ sim_length: {self.sim_length} steps\n"
            f"└─ frequency:  {self.freq}"
        )