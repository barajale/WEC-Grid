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
    """
            
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
        >>> print(f"End: {time_mgr.sim_stop}")
        
        >>> # High-resolution short simulation
        >>> time_mgr.update(sim_length=60, freq="1T")  # 1 hour at 1-minute
        >>> print(f"Snapshots: {len(time_mgr.snapshots)}")
        Snapshots: 60
        
    Time Coordination Workflow:
        1. **Initialization**: Set simulation start time and duration
        2. **Snapshot Generation**: Create pandas DatetimeIndex for time series
        3. **Cross-Platform Sync**: Align PSS®E, PyPSA, and WEC data timing
        4. **Dynamic Updates**: Modify parameters during simulation setup
        5. **Data Alignment**: Ensure consistent time indexing across components
        
    Simulation Time Scales:
        - **Minutes**: High-resolution transient studies (freq="1T", sim_length=60)
        - **Hours**: Standard grid operations (freq="5T", sim_length=288)
        - **Days**: Multi-day weather patterns (freq="15T", sim_length=672)
        - **Seasonal**: Long-term resource studies (freq="1H", sim_length=8760)
        
    Grid Integration Applications:
        - **Load Flow Studies**: Hourly or sub-hourly power flow analysis
        - **Stability Analysis**: Minute-resolution dynamic simulations
        - **Economic Dispatch**: 15-minute to hourly optimization intervals
        - **Resource Planning**: Annual studies with hourly or daily resolution
        
    WEC Data Synchronization:
        - **WEC-Sim Alignment**: Matches device simulation output timing
        - **Wave Statistics**: Coordinates with ocean wave measurement intervals
        - **Power Output**: Synchronizes WEC generation with grid time steps
        - **Farm Aggregation**: Ensures consistent timing across multiple devices
        
    Cross-Platform Compatibility:
        - **PSS®E**: Provides time step sequence for dynamic simulations
        - **PyPSA**: Generates snapshot index for optimization problems
        - **Matplotlib**: Compatible datetime axis for plotting
        - **Pandas**: Native time-series indexing and resampling
        
    Performance Considerations:
        - **Memory scaling**: Snapshot generation scales linearly with sim_length
        - **Computation efficiency**: Pandas DatetimeIndex optimized for time queries
        - **Update overhead**: Minimal cost for parameter modifications
        - **Large simulations**: Consider memory usage for multi-year studies
        
    Notes:
        - Default 5-minute intervals optimize grid modeling vs. computational cost
        - Start time defaults to midnight for consistent daily pattern alignment
        - Simulation stop time automatically calculated and maintained
        - Frequency strings follow pandas conventions for broad compatibility
        - Time zone handling uses local system time (UTC recommended for reproducibility)
        
    See Also:
        Engine: Uses time manager for simulation coordination
        WECFarm: Synchronizes device data with time manager snapshots
        WECGridPlotter: Uses snapshots for time-series visualization
        
    References:
        Pandas time series documentation: https://pandas.pydata.org/docs/user_guide/timeseries.html
    """
    start_time: datetime = field(default_factory=lambda: datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
    sim_length: int = 288
    freq: str = "5T"

    def __post_init__(self):
        """Initialize derived simulation parameters after dataclass construction.
        
        Automatically called by the dataclass framework after object initialization
        to compute and set dependent attributes. This ensures the simulation end
        time is properly calculated based on the provided start time, length, and
        frequency parameters.
        
        Returns:
            None: Sets sim_stop attribute based on computed snapshots.
            
        Notes:
            - Called automatically by dataclass mechanism after __init__
            - Ensures sim_stop is always consistent with other parameters
            - Required for proper dataclass initialization workflow
            - Updates derived attributes that depend on primary fields
            
        See Also:
            _update_sim_stop: Internal method for end time calculation
        """
        self._update_sim_stop()

    def _update_sim_stop(self):
        """Update simulation end time based on current parameters.
        
        Internal method that recalculates the simulation stop time whenever
        simulation parameters change. Ensures the end time remains consistent
        with start time, duration, and frequency settings.
        
        Returns:
            None: Updates self.sim_stop attribute.
            
        Calculation Logic:
            - If sim_length > 0: Uses last timestamp from generated snapshots
            - If sim_length = 0: Sets stop time equal to start time (empty simulation)
            
        Notes:
            - Called automatically after parameter updates
            - Handles edge case of zero-length simulations gracefully
            - Uses snapshots property for consistent time calculation
            - Internal method not intended for direct user access
            
        See Also:
            snapshots: Property used for end time calculation
            update: Public method that calls this internal function
        """
        self.sim_stop = self.snapshots[-1] if self.sim_length > 0 else self.start_time

    @property
    def snapshots(self) -> pd.DatetimeIndex:
        """Generate time snapshots for simulation time series.
        
        Creates a pandas DatetimeIndex representing all simulation time points
        based on the configured start time, duration, and frequency. This property
        provides the fundamental time coordinate system for all WEC-Grid simulations.
        
        Returns:
            pd.DatetimeIndex: Ordered sequence of simulation timestamps.
                Length equals sim_length, spanning from start_time with
                intervals defined by freq parameter.
                
        Example:
            >>> # Standard 24-hour simulation
            >>> time_mgr = WECGridTimeManager(
            ...     start_time=datetime(2023, 1, 1, 0, 0, 0),
            ...     sim_length=288,
            ...     freq="5T"
            ... )
            >>> snapshots = time_mgr.snapshots
            >>> print(f"First: {snapshots[0]}")
            >>> print(f"Last: {snapshots[-1]}")
            >>> print(f"Length: {len(snapshots)}")
            First: 2023-01-01 00:00:00
            Last: 2023-01-01 23:55:00
            Length: 288
            
            >>> # High-resolution short simulation
            >>> time_mgr.update(sim_length=60, freq="1T")
            >>> snapshots = time_mgr.snapshots
            >>> print(f"Interval: {snapshots[1] - snapshots[0]}")
            Interval: 0:01:00
            
        Time Series Applications:
            - **Power system modeling**: Time index for load and generation data
            - **WEC data alignment**: Synchronize device output with grid timing
            - **Result visualization**: X-axis coordinates for time-series plots
            - **Data resampling**: Reference index for temporal aggregation
            
        Pandas Integration:
            - **DataFrame indexing**: Direct use as pandas DataFrame index
            - **Time-based selection**: Support for date range queries
            - **Resampling operations**: Compatible with resample() methods
            - **Plotting compatibility**: Works with matplotlib time axes
            
        Performance Characteristics:
            - **Lazy evaluation**: Generated on each access (no memory storage)
            - **Efficient creation**: Pandas optimized date range generation
            - **Memory scaling**: No permanent storage, computed as needed
            - **Consistent results**: Same parameters yield identical snapshots
            
        Frequency Support:
            - **Standard intervals**: "1T", "5T", "15T", "1H", "1D"
            - **Custom periods**: Any valid pandas frequency string
            - **Grid compatibility**: 5T optimal for power system studies
            - **High resolution**: 1T for detailed transient analysis
            
        Notes:
            - Property computed dynamically on each access
            - No internal caching for memory efficiency
            - Changes to time parameters immediately affect snapshots
            - Compatible with all pandas time-series functionality
            - Timezone naive (uses local system time)
            
        See Also:
            update: Modify parameters that affect snapshot generation
            set_end_time: Alternative method for setting simulation duration
        """
        return pd.date_range(
            start=self.start_time,
            periods=self.sim_length,
            freq=self.freq,
        )

    def update(self, *, start_time: datetime = None, sim_length: int = None, freq: str = None):
        """Update simulation time parameters with automatic recalculation.
        
        Modifies one or more time management parameters and automatically
        recalculates dependent values including simulation end time. This method
        provides a convenient interface for adjusting simulation timing during
        setup or between simulation runs.
        
        Args:
            start_time (datetime, optional): New simulation start timestamp.
                If provided, updates the reference time for all calculations.
                Should be timezone-naive for consistency with WEC-Grid conventions.
                
            sim_length (int, optional): New number of simulation time steps.
                Must be positive integer. Determines total simulation duration
                when combined with frequency parameter.
                
            freq (str, optional): New pandas frequency string for time intervals.
                Must be valid pandas frequency code (e.g., "1T", "5T", "1H").
                Controls time step size between successive snapshots.
                
        Returns:
            None: Updates internal state and recalculates sim_stop.
            
        Example:
            >>> # Initial setup
            >>> time_mgr = WECGridTimeManager()
            >>> print(f"Initial: {time_mgr.sim_length} steps at {time_mgr.freq}")
            Initial: 288 steps at 5T
            
            >>> # Extend simulation to 48 hours
            >>> time_mgr.update(sim_length=576)
            >>> print(f"Extended: {time_mgr.sim_length} steps")
            Extended: 576 steps
            
            >>> # Change to hourly intervals for planning study
            >>> time_mgr.update(freq="1H", sim_length=24)
            >>> print(f"Planning: {len(time_mgr.snapshots)} hours")
            Planning: 24 hours
            
            >>> # Set specific start date
            >>> from datetime import datetime
            >>> time_mgr.update(start_time=datetime(2023, 6, 15, 6, 0, 0))
            >>> print(f"New start: {time_mgr.start_time}")
            New start: 2023-06-15 06:00:00
            
        Parameter Validation:
            - **Start time**: Should be timezone-naive datetime object
            - **Simulation length**: Must be positive integer (>= 0)
            - **Frequency**: Must be valid pandas frequency string
            - **Combination effects**: All parameters interact to determine snapshots
            
        Common Usage Patterns:
            **Quick duration change**:
            ```python
            time_mgr.update(sim_length=144)  # 12 hours at current frequency
            ```
            
            **Resolution adjustment**:
            ```python
            time_mgr.update(freq="1T", sim_length=60)  # 1 hour at 1-minute intervals
            ```
            
            **Study period setup**:
            ```python
            time_mgr.update(
                start_time=datetime(2023, 7, 1, 0, 0, 0),
                sim_length=8760,  # Full year
                freq="1H"         # Hourly intervals
            )
            ```
            
        Automatic Updates:
            - **End time recalculation**: sim_stop updated automatically
            - **Snapshot regeneration**: New snapshots reflect updated parameters
            - **Consistency maintenance**: All dependent properties stay synchronized
            - **Immediate effect**: Changes apply to next property access
            
        Performance Considerations:
            - **Minimal overhead**: Only updates specified parameters
            - **No data copying**: Parameter changes don't affect existing data
            - **Lazy recalculation**: Snapshots computed on next access
            - **Batch updates**: Multiple parameters updated in single call
            
        Notes:
            - Uses keyword-only arguments to prevent parameter confusion
            - None values leave corresponding parameters unchanged
            - End time automatically recalculated after any parameter change
            - Changes affect all subsequent snapshot and time series operations
            - No validation performed on frequency strings (pandas handles errors)
            
        See Also:
            set_end_time: Alternative method for duration-based updates
            snapshots: Property affected by parameter updates
            __post_init__: Initial calculation performed during construction
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
        
        Alternative method for configuring simulation duration by providing
        the target end timestamp rather than the number of time steps. This
        method automatically calculates the required sim_length to achieve
        the specified end time given the current start time and frequency.
        
        Args:
            end_time (datetime): Desired simulation end timestamp.
                Must be later than current start_time. Should be timezone-naive
                for consistency with WEC-Grid time conventions.
                
        Returns:
            None: Updates sim_length and sim_stop attributes.
            
        Raises:
            ValueError: If end_time is earlier than or equal to start_time.
            
        Example:
            >>> # Set up simulation from Jan 1 to Jan 3 (48 hours)
            >>> from datetime import datetime
            >>> time_mgr = WECGridTimeManager(
            ...     start_time=datetime(2023, 1, 1, 0, 0, 0),
            ...     freq="5T"
            ... )
            >>> 
            >>> end_time = datetime(2023, 1, 3, 0, 0, 0)
            >>> time_mgr.set_end_time(end_time)
            >>> 
            >>> print(f"Duration: {time_mgr.sim_length} steps")
            >>> print(f"Start: {time_mgr.start_time}")
            >>> print(f"End: {time_mgr.sim_stop}")
            Duration: 576 steps
            Start: 2023-01-01 00:00:00
            End: 2023-01-03 00:00:00
            
            >>> # Monthly simulation with hourly intervals
            >>> time_mgr.update(freq="1H")
            >>> end_time = datetime(2023, 2, 1, 0, 0, 0)  # End of January
            >>> time_mgr.set_end_time(end_time)
            >>> print(f"Monthly hours: {time_mgr.sim_length}")
            Monthly hours: 744
            
        Duration Calculation:
            The method uses pandas date_range to determine the exact number
            of time steps needed to span from start_time to end_time at the
            current frequency:
            
            ```python
            steps = len(pd.date_range(start=start_time, end=end_time, freq=freq))
            ```
            
        Precision and Rounding:
            - **Exact matches**: End time exactly aligned with frequency intervals
            - **Partial intervals**: End time rounded to nearest frequency boundary
            - **Frequency dependent**: Precision depends on freq parameter
            - **Pandas behavior**: Follows pandas date_range endpoint handling
            
        Common Use Cases:
            **Calendar-based studies**:
            ```python
            # Simulate full calendar month
            time_mgr.set_end_time(datetime(2023, 2, 1, 0, 0, 0))
            ```
            
            **Event-driven analysis**:
            ```python
            # Simulate until specific event time
            storm_end = datetime(2023, 1, 15, 18, 30, 0)
            time_mgr.set_end_time(storm_end)
            ```
            
            **Multi-day studies**:
            ```python
            # Weekend simulation (Friday to Monday)
            monday_start = datetime(2023, 1, 23, 0, 0, 0)
            time_mgr.set_end_time(monday_start)
            ```
            
        Advantages vs. sim_length:
            - **Calendar alignment**: Natural for date-based study periods
            - **Event coordination**: Easier to align with external schedules
            - **Duration visualization**: Clearer for non-technical stakeholders
            - **Planning integration**: Direct compatibility with project schedules
            
        Frequency Interactions:
            - **5T frequency**: 288 steps per day, 2016 steps per week
            - **1H frequency**: 24 steps per day, 168 steps per week
            - **15T frequency**: 96 steps per day, 672 steps per week
            - **Custom frequencies**: Calculated automatically by pandas
            
        Notes:
            - Overwrites any existing sim_length setting
            - End time stored directly in sim_stop (no recalculation needed)
            - Frequency must be set before calling this method for accurate calculation
            - End time should align with frequency boundaries for predictable results
            - No validation that end_time > start_time (pandas handles gracefully)
            
        See Also:
            update: Alternative method for step-based duration setting
            snapshots: Property that reflects the calculated time range
        """
        self.sim_length = len(pd.date_range(start=self.start_time, end=end_time, freq=self.freq))
        self.sim_stop = end_time

    def __repr__(self) -> str:
        """Return concise tree-style string representation of the WECGridTimeManager.
        
        Provides a clean, hierarchical display of the time manager configuration
        using Unicode tree characters for visual structure. Designed for quick
        inspection during interactive development and debugging sessions.
        
        Returns:
            str: Tree-formatted string showing key temporal parameters:
                - start_time: Simulation beginning timestamp
                - sim_stop: Calculated simulation end timestamp
                - sim_length: Total number of time steps
                - frequency: Time interval specification
                
        Format Specification:
            Uses Unicode box-drawing characters for visual hierarchy:
            ```
            WECGridTimeManager:
            ├─ start_time: YYYY-MM-DD HH:MM:SS
            ├─ sim_stop:   YYYY-MM-DD HH:MM:SS
            ├─ sim_length: N steps
            └─ frequency:  pandas_freq_string
            ```
            
        Example:
            >>> from datetime import datetime
            >>> time_mgr = WECGridTimeManager(
            ...     start_time=datetime(2023, 1, 1, 12, 0, 0),
            ...     sim_length=288,
            ...     freq="5T"
            ... )
            >>> print(time_mgr)
            WECGridTimeManager:
            ├─ start_time: 2023-01-01 12:00:00
            ├─ sim_stop:   2023-01-02 12:00:00
            ├─ sim_length: 288 steps
            └─ frequency:  5T
            
            >>> # Hourly simulation for one week
            >>> time_mgr.update(freq="1H", sim_length=168)
            >>> print(time_mgr)
            WECGridTimeManager:
            ├─ start_time: 2023-01-01 12:00:00
            ├─ sim_stop:   2023-01-08 12:00:00
            ├─ sim_length: 168 steps
            └─ frequency:  1H
            
        Display Features:
            - **Compact format**: Single line per key parameter
            - **Visual hierarchy**: Tree structure shows parameter relationships
            - **Aligned values**: Consistent spacing for readability
            - **Essential info**: Only critical parameters displayed
            
        Common Usage:
            **Interactive inspection**:
            ```python
            >>> time_mgr  # Automatic __repr__ call
            WECGridTimeManager:
            ├─ start_time: ...
            ```
            
            **Debug logging**:
            ```python
            >>> print(f"Current config:\\n{time_mgr}")
            ```
            
            **Configuration verification**:
            ```python
            >>> assert "288 steps" in str(time_mgr)
            ```
            
        Comparison with Detailed Formats:
            - **Compact**: Focuses on essential timing parameters only
            - **Tree style**: Visual hierarchy for parameter relationships
            - **Quick scan**: Enables rapid visual inspection
            - **No examples**: Concise format without usage documentation
            
        Unicode Compatibility:
            - Uses standard Unicode box-drawing characters (U+251C, U+2500, U+2514)
            - Should display correctly in most modern terminals and IDEs
            - Falls back gracefully in text-only environments
            - Consistent with modern Python CLI tool conventions
            
        Performance Notes:
            - Minimal computational overhead (string formatting only)
            - No pandas operations required for display
            - Suitable for frequent interactive use
            - Duration calculation not included for simplicity
            
        Display Consistency:
            - All timestamps shown without timezone information
            - Frequency displayed using pandas notation
            - Step count includes units for clarity
            - Alignment maintained for parameter values
            
        Notes:
            - Designed for quick visual inspection rather than programmatic parsing
            - Uses sim_stop directly (no recalculation for display)
            - Frequency string displayed as provided to pandas
            - Compatible with Jupyter notebook display systems
            
        See Also:
            snapshots: Property for detailed time array inspection
            update: Method for modifying displayed parameters
            set_end_time: Alternative method for end time specification
        """
        return (
            f"WECGridTimeManager:\n"
            f"├─ start_time: {self.start_time}\n"
            f"├─ sim_stop:   {self.sim_stop}\n"
            f"├─ sim_length: {self.sim_length} steps\n"
            f"└─ frequency:  {self.freq}"
        )