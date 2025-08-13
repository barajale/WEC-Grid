# File: src/wecgrid/util/wecgrid_timemanager.py

"""Time management and coordination for WEC-Grid simulations.

Provides the WECGridTimeManager dataclass for coordinating simulation time
across WEC-Grid components including power system modeling, WEC device
simulations, and data visualization with consistent temporal alignment.
"""

from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

@dataclass
class WECGridTimeManager:
    """Centralized time coordination for WEC-Grid simulations.
    
    Coordinates temporal aspects across power system modeling (PSS®E, PyPSA),
    WEC simulations (WEC-Sim), and visualization components. Manages simulation
    time windows, sampling intervals, and ensures cross-platform alignment.
    
    Attributes:
        start_time (datetime): Simulation start timestamp. Defaults to current 
            date at midnight.
        sim_length (int): Number of simulation time steps. Defaults to 288
            (24 hours at 5-minute intervals).
        freq (str): Pandas frequency string for time intervals. Defaults to "5T"
            (5-minute intervals).
            
        sim_stop (datetime): Calculated simulation end timestamp.
            Automatically computed from start_time, sim_length, and freq.
            Updated whenever simulation parameters change.
            
    Example:
        >>> # Default 24-hour simulation at 5-minute intervals
        >>> time_mgr = WECGridTimeManager()
        >>> print(f"Duration: {time_mgr.sim_length} steps")
        >>> print(f"Interval: {time_mgr.freq}")
        Duration: 288 steps
        Interval: 5T
        
        >>> # Custom simulation period
        >>> from datetime import datetime
        >>> time_mgr = WECGridTimeManager(
        ...     start_time=datetime(2023, 6, 15, 0, 0, 0),
        ...     sim_length=144,  # 12 hours
        ...     freq="5T"
        ... )
        >>> print(f"Start: {time_mgr.start_time}")
    """
    start_time: datetime = field(default_factory=lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    sim_length: int = 288
    freq: str = "5T"

    def __post_init__(self):
        """Initialize derived simulation parameters after dataclass construction."""
        self._update_sim_stop()

    def _update_sim_stop(self):
        """Update simulation stop time based on current parameters."""
        self.sim_stop = self.snapshots[-1] if self.sim_length > 0 else self.start_time

    @property
    def snapshots(self) -> pd.DatetimeIndex:
        """Generate time snapshots for simulation time series.
        
        Returns:
            pd.DatetimeIndex: Simulation timestamps from start_time with length sim_length.
        """
        return pd.date_range(
            start=self.start_time,
            periods=self.sim_length,
            freq=self.freq,
        )

    def update(self, *, start_time: datetime = None, sim_length: int = None, freq: str = None):
        """Update simulation time parameters with automatic recalculation.
        
        Args:
            start_time (datetime, optional): New simulation start timestamp.
            sim_length (int, optional): New number of simulation time steps.
            freq (str, optional): New pandas frequency string for time intervals.
        """
        if start_time is not None:
            self.start_time = start_time
        if sim_length is not None:
            self.sim_length = sim_length
        if freq is not None:
            self.freq = freq
        self._update_sim_stop()

    def set_end_time(self, end_time: datetime):
        """Set simulation duration by specifying the desired end time.
        
        Args:
            end_time (datetime): Desired simulation end timestamp.
                Must be later than current start_time.
                
        Raises:
            ValueError: If end_time is earlier than or equal to start_time.
        """
        self.sim_length = len(pd.date_range(start=self.start_time, end=end_time, freq=self.freq))
        self.sim_stop = end_time

    def __repr__(self) -> str:
        """Return concise string representation of the WECGridTimeManager."""
        return (
            f"WECGridTimeManager:\n"
            f"├─ start_time: {self.start_time}\n"
            f"├─ sim_stop:   {self.sim_stop}\n"
            f"├─ sim_length: {self.sim_length} steps\n"
            f"└─ frequency:  {self.freq}"
        )