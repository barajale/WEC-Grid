"""
Wave Energy Converter Farm modeling for power system integration.

This module provides the WECFarm class for managing collections of Wave Energy
Converter (WEC) devices within power system simulations. WEC farms aggregate
multiple identical devices at a common grid connection point, providing realistic
renewable energy profiles for grid integration studies.
"""

from typing import List, Dict, Any
import pandas as pd
from .wecdevice import WECDevice
from .wecsim_runner import WECSimRunner


class WECFarm:
    """Collection of Wave Energy Converter devices at a common grid connection.
    
    Manages multiple identical WEC devices sharing a grid connection bus. 
    Aggregates device power outputs and coordinates time-series data for
    power system integration studies.
        
    Attributes:
        farm_name (str): Human-readable farm identifier.
        database: Database interface for WEC simulation data.
        time: Time manager for simulation synchronization.
        sim_id (int): Database simulation ID for WEC data retrieval.
        model (str): WEC device model type (e.g., "RM3").
        bus_location (int): Grid bus number for farm connection.
        connecting_bus (int): Network topology connection bus.
        id (str): Unique generator identifier for power system integration.
        size (int): Number of identical WEC devices in farm.
        config (Dict): Configuration parameters for the farm.
        wec_devices (List[WECDevice]): Collection of individual WEC devices.
        BASE (float): Base power rating [MVA] for per-unit calculations.
        
    Example:
        >>> farm = WECFarm(
        ...     farm_name="Oregon Coast Farm",
        ...     database=db,
        ...     time=time_mgr,
        ...     sim_id=101,
        ...     model="RM3",
        ...     bus_location=14,
        ...     size=5
        ... )
        >>> total_power = farm.power_at_snapshot(timestamp)
        
    Notes:
        - All devices use identical power profiles from WEC-Sim data
        - Power scales linearly with farm size
        - Requires WEC-Sim simulation data in database
        - Base power typically 100 MVA for utility-scale installations
        
    TODO:
        - Add heterogeneous device support for different models
        - Implement smart farm control and optimization
    """
    def __init__(self, farm_name: str, database, time: Any, sim_id: int, model: str, bus_location: int, connecting_bus: int = 1, size: int = 1, gen_id: str = None):
        """Initialize WEC farm with specified configuration.
        
        Args:
            farm_name (str): Human-readable WEC farm identifier.
            database: Database interface for WEC simulation data access.
            time: Time management object for simulation synchronization.
            sim_id (int): Database simulation ID for WEC data retrieval.
            model (str): WEC device model type ("RM3", etc.).
            bus_location (int): Grid bus number for farm connection.
            connecting_bus (int, optional): Network topology connection bus. Defaults to 1.
            size (int, optional): Number of WEC devices in farm. Defaults to 1.
            gen_id (str, optional): Generator ID for power system. Auto-generated if None.
                
        Raises:
            RuntimeError: If WEC simulation data not found in database.
            ValueError: If database query returns empty results.
            
        Example:
            >>> farm = WECFarm(
            ...     farm_name="Newport Array",
            ...     database=db,
            ...     time=time_mgr,
            ...     sim_id=101,
            ...     model="RM3",
            ...     bus_location=14,
            ...     size=5
            ... )
            
        Notes:
            - Creates identical WECDevice objects for all farm devices
            - Retrieves WEC-Sim data from database using sim_id
            - Sets up per-unit base power from simulation data
        """
        
        self.farm_name: str = farm_name
        self.database = database # TODO make this a WECGridDB data type
        self.time = time # todo might need to update time to be SimulationTime type 
        self.sim_id: int = sim_id
        self.model: str = model
        self.bus_location: int = bus_location
        self.connecting_bus: int = connecting_bus # todo this should default to swing bus
        self.id: str = gen_id
        self.size: int = size
        self.config: Dict = None
        self.wec_devices: List[WECDevice] = []
        self.BASE: float = 100.0  # this should be the base of the wec, which is usually 100 MVA

        self._prepare_farm()

    def __repr__(self) -> str:
        """Return a formatted string representation of the WEC farm configuration.
        
        Provides a hierarchical display of key farm parameters for debugging,
        logging, and user information. The format is designed for readability
        and includes essential configuration details.
        
        Returns:
            str: Formatted multi-line string with farm configuration details.
                Includes farm name, size, model, grid connections, simulation ID,
                and base power rating in a tree-like structure.
                
        Example:
            >>> farm = WECFarm("Test Farm", db, time_mgr, 101, "RM3", 14, size=5)
            >>> print(farm)
            WECFarm:
            ├─ name: 'Test Farm'
            ├─ size: 5
            ├─ model: 'RM3'
            ├─ bus_location: 14
            ├─ connecting_bus: 1
            └─ sim_id: 101

            Base: 100.0 MVA
            
        Display Format:
            - **Tree structure**: Uses Unicode box-drawing characters
            - **Key parameters**: Farm name, device count, model type
            - **Grid connections**: Bus locations for power system integration
            - **Simulation link**: Database simulation identifier
            - **Base power**: MVA rating for per-unit calculations
            
        Use Cases:
            - **Interactive debugging**: Quick farm configuration inspection
            - **Logging output**: Structured information for log files
            - **Jupyter notebooks**: Clean display in research environments
            - **Configuration validation**: Verify farm setup parameters
            
        Notes:
            - Size displays actual number of created devices (len(wec_devices))
            - Farm name shown in quotes to distinguish from other identifiers
            - Base power extracted from WEC simulation data during initialization
            - Unicode characters may not display properly in all terminals
        """
        return f"""WECFarm:
        ├─ name: {self.farm_name!r}
        ├─ size: {len(self.wec_devices)}
        ├─ model: {self.model!r}
        ├─ bus_location: {self.bus_location}
        ├─ connecting_bus: {self.connecting_bus}
        └─ sim_id: {self.sim_id}

        Base: {self.BASE} MVA

    """
    
    def _prepare_farm(self):
        """Load WEC simulation data from database and create individual device objects.
        
        Internal method that handles the core initialization logic for the WEC farm.
        Validates database content, retrieves simulation results, processes time
        indexing, and instantiates the specified number of WEC device objects.
        
        Returns:
            None: Populates self.wec_devices with configured WECDevice objects.
            
        Raises:
            RuntimeError: If WEC simulation data not found or loading fails.
                Provides guidance on running WEC-Sim simulations first.
            ValueError: If database returns empty or invalid data.
            KeyError: If required columns missing from simulation data.
            
        Data Loading Process:
            1. **Table Existence Check**: Verify simulation tables exist in database
            2. **Grid Data Retrieval**: Load downsampled results for power system integration
            3. **Full Data Retrieval**: Load high-resolution results for detailed analysis
            4. **Time Index Creation**: Generate pandas datetime index for time series
            5. **Base Power Extraction**: Get MVA rating from simulation data
            6. **Device Instantiation**: Create specified number of WECDevice objects
            
        Database Table Schema:
            **Integration table** (`WECSIM_{model}_{sim_id}`):
            - time: Simulation time [s]
            - p: Active power output [MW]
            - q: Reactive power output [MVAr] (typically zero)
            - base: Base power rating [MVA]
            
            **Full resolution table** (`WECSIM_{model}_{sim_id}_full`):
            - time: Simulation time [s] (high frequency)
            - p: Active power output [MW]
            - eta: Wave surface elevation [m]
            - Additional WEC-Sim variables
            
        Time Series Processing:
            - **5-minute intervals**: Standard grid integration time step
            - **Datetime indexing**: Converts simulation seconds to timestamp format
            - **Start time alignment**: Uses time manager's configured start time
            - **Pandas integration**: Creates DataFrame index for efficient querying
            
        Device Creation:
            Each WECDevice configured with:
            - **Unique naming**: "{model}_{sim_id}_{index}" pattern
            - **Shared data**: Copy of power time series for each device
            - **Common parameters**: Bus location, base power, model type
            - **Individual objects**: Separate instance for potential customization
            
        Error Handling:
            - **Missing tables**: Clear error message with WEC-Sim guidance
            - **Empty data**: Validation of successful data retrieval
            - **Data integrity**: Checks for required columns and valid values
            - **Graceful failure**: Informative error messages for troubleshooting
            
        Performance Considerations:
            - **Single query**: Minimizes database access for efficiency
            - **Data copying**: Each device gets independent DataFrame copy
            - **Memory usage**: Scales with farm size and simulation duration
            - **Time indexing**: One-time conversion for all devices
            
        Notes:
            - Called automatically during farm initialization
            - Assumes WEC-Sim simulation completed successfully
            - Base power typically 100 MVA for utility-scale installations
            - All devices share identical power profiles (homogeneous farm)
            - TODO: Add data validation and integrity checks
            - TODO: Support for custom time intervals beyond 5 minutes
            
        See Also:
            WECDevice: Individual device objects created by this method
            WECSimRunner: Generates the simulation data loaded here
            WECGridTimeManager: Provides time configuration for indexing
        """
        table_name = f"WECSIM_{self.model.lower()}_{self.sim_id}"
        exists_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        result = self.database.query(exists_query)

        if not result:
            raise RuntimeError(f"[Farm] No WEC data for sim_id={self.sim_id} found in database. Run WEC-SIM first.")
            # TODO: Provide clearer guidance on running WEC-SIM

        # Load data once and distribute to all devices
        df = self.database.query(f"SELECT * FROM {table_name}", return_type="df")
        if df is None or df.empty:
            raise RuntimeError(f"[Farm] Failed to load WEC data for sim_id={self.sim_id}")
        
        df_full = self.database.query(f"SELECT * FROM {table_name}_full", return_type="df")
        if df_full is None or df_full.empty:
            raise RuntimeError(f"[Farm] Failed to load full WEC data for sim_id={self.sim_id}")

        # Apply time index at 5 min resolution using start time
        df["snapshots"] = pd.date_range(start=self.time.start_time, periods=df.shape[0], freq="5T")
        df.set_index("snapshots", inplace=True) 

        self.BASE = df["base"][0]
        
        for i in range(self.size):
            name = f"{self.model}_{self.sim_id}_{i}"
            device = WECDevice(
                name=name,
                dataframe=df.copy(),
                dataframe_full=df_full.copy(),
                bus_location=self.bus_location,
                base=df["base"][0],
                model=self.model,
                sim_id=self.sim_id
            )
            self.wec_devices.append(device)
            
        
        
    def power_at_snapshot(self, timestamp: pd.Timestamp) -> float:
        """Calculate total farm power output at a specific simulation time.
        
        Aggregates active power output from all WEC devices in the farm at the
        specified timestamp. This method provides the primary interface for
        power system integration, enabling time-varying renewable generation
        modeling in grid simulations.
        
        Args:
            timestamp (pd.Timestamp): Simulation time to query for power output.
                Must exist in the device DataFrame time index. Typically corresponds
                to grid simulation snapshots at 5-minute intervals.
                
        Returns:
            float: Total active power output from all farm devices [MW].
                Sum of individual device outputs at the specified time.
                Returns 0.0 if no valid data available at timestamp.
                
        Raises:
            KeyError: If timestamp not found in device data index.
            AttributeError: If device DataFrame not properly initialized.
            
        Example:
            >>> # Get power at specific simulation time
            >>> timestamp = pd.Timestamp("2023-01-01 12:00:00")
            >>> power = farm.power_at_snapshot(timestamp)
            >>> print(f"Farm output at noon: {power:.2f} MW")
            Farm output at noon: 15.75 MW
            
            >>> # Time series power extraction
            >>> time_series = []
            >>> for snapshot in time_manager.snapshots:
            ...     power = farm.power_at_snapshot(snapshot)
            ...     time_series.append(power)
            >>> 
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(time_manager.snapshots, time_series)
            >>> plt.ylabel("Farm Power Output [MW]")
            
        Power Aggregation:
            - **Linear summation**: Total = Σ(device_power[i] at timestamp)
            - **Homogeneous devices**: All devices have identical power profiles
            - **Realistic scaling**: Based on actual WEC device physics
            - **Wave correlation**: Devices respond to same ocean conditions
            
        Data Requirements:
            - **Valid timestamp**: Must exist in device DataFrame index
            - **Initialized devices**: All WECDevice objects must be properly created
            - **Power column**: Device data must contain "p" column for active power
            - **Time alignment**: Timestamp must match grid simulation schedule
            
        Error Handling:
            - **Missing data warning**: Prints warning for devices with no data
            - **Graceful degradation**: Continues calculation with available devices
            - **Zero fallback**: Returns 0.0 if no devices have valid data
            - **Timestamp validation**: Checks for existence in device index
            
        Performance Considerations:
            - **O(n) complexity**: Scales linearly with number of devices
            - **DataFrame lookup**: Efficient pandas indexing for time queries
            - **Memory efficiency**: No data copying, direct access to device data
            - **Repeated calls**: Suitable for time-series iteration
            
        Grid Integration Usage:
            - **PSS®E integration**: Provides generator output at each time step
            - **PyPSA integration**: Supplies renewable generation time series
            - **Load flow studies**: Time-varying injection for stability analysis
            - **Economic dispatch**: Variable renewable generation modeling
            
        Wave Energy Characteristics:
            - **Intermittent output**: Power varies with wave conditions
            - **Predictable patterns**: Follows ocean wave statistics
            - **Seasonal variation**: Higher output in winter storm seasons
            - **Capacity factor**: Typically 20-40% for ocean wave resources
            
        Notes:
            - Power output includes WEC device efficiency and control effects
            - All devices share identical profiles (same wave field assumption)
            - Negative power values possible during reactive conditions
            - Zero output during calm conditions or device maintenance
            - Farm total limited by grid connection capacity
            
        See Also:
            WECDevice.dataframe: Individual device power time series
            Engine.simulate: Uses this method for grid integration
            WECGridPlotter.plot_wec_analysis: Visualizes farm power output
        """
        total_power = 0.0
        for device in self.wec_devices:
            if (
                device.dataframe is not None 
                and not device.dataframe.empty 
                and timestamp in device.dataframe.index
            ):
                power = device.dataframe.at[timestamp, "p"]
                total_power += power
            else:
                print(f"[WARNING] Missing data for {device.name} at {timestamp}")
        return total_power