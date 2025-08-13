"""
Individual Wave Energy Converter device modeling for power system integration.

This module provides the WECDevice dataclass for representing individual Wave Energy
Converter (WEC) devices with their associated time-series data, grid connection
parameters, and simulation metadata. WECDevice objects form the building blocks
of WEC farms in power system studies.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import pandas as pd
from datetime import datetime

@dataclass
class WECDevice:
    """Individual Wave Energy Converter device with time-series power output data.
    
    The WECDevice dataclass represents a single wave energy converter with its
    associated simulation results, grid connection parameters, and metadata.
    Each device contains time-series power output data derived from high-fidelity
    WEC-Sim hydrodynamic simulations, enabling realistic renewable generation
    modeling in power system studies.
    
    This dataclass serves as the fundamental unit for WEC farm aggregation and
    provides the interface between device-level wave energy simulations and
    grid-scale power system analysis.
    
    Attributes:
        name (str): Unique identifier for the WEC device.
            Typically formatted as "{model}_{sim_id}_{index}" for farm devices.
            Used for device tracking, logging, and result identification.
            
        dataframe (pd.DataFrame): Primary time-series data for grid integration.
            Contains downsampled simulation results at 5-minute intervals.
            Standard columns:
            - time: Simulation time [s] (converted to datetime index)
            - p: Active power output [MW]
            - q: Reactive power output [MVAr] (typically zero for WECs)
            - base: Base power rating [MVA]
            
        dataframe_full (pd.DataFrame): High-resolution simulation data.
            Contains complete WEC-Sim output at original time resolution.
            Additional columns may include:
            - eta: Wave surface elevation [m]
            - Forces and motions from WEC-Sim simulation
            - Device-specific state variables
            
        base (float, optional): Base power rating in MVA for per-unit calculations.
            Typically 100 MVA for utility-scale installations.
            Used for normalizing power values in power system software.
            Extracted automatically from WEC-Sim simulation results.
            
        bus_location (int, optional): Power system bus number for grid connection.
            Must correspond to valid bus in the power system case file.
            Used by PSS®E and PyPSA for electrical network modeling.
            
        model (str, optional): WEC device model type identifier.
            Examples: "RM3", "LUPA", "OSWEC", custom models.
            Links device to specific hydrodynamic and control characteristics.
            
        sim_id (int, optional): Database simulation identifier.
            References the WEC-Sim simulation run used for this device.
            Enables traceability between device data and simulation parameters.
            
    Example:
        >>> # Create device from WEC-Sim simulation results
        >>> import pandas as pd
        >>> 
        >>> # Sample time-series data
        >>> time_data = pd.date_range("2023-01-01", periods=288, freq="5T")
        >>> power_data = pd.DataFrame({
        ...     'p': [2.5, 3.1, 2.8, 3.4, 2.9],  # MW
        ...     'q': [0.0, 0.0, 0.0, 0.0, 0.0],  # MVAr
        ...     'base': [100.0] * 5               # MVA
        ... }, index=time_data[:5])
        >>> 
        >>> device = WECDevice(
        ...     name="RM3_101_0",
        ...     dataframe=power_data,
        ...     base=100.0,
        ...     bus_location=14,
        ...     model="RM3",
        ...     sim_id=101
        ... )
        >>> 
        >>> print(f"Device: {device.name}")
        >>> print(f"Power at first timestep: {device.dataframe.iloc[0]['p']:.1f} MW")
        Device: RM3_101_0
        Power at first timestep: 2.5 MW
        
    Device Characteristics:
        - **Renewable generation**: Variable power output based on wave conditions
        - **Zero reactive power**: Most WEC devices operate at unity power factor
        - **Time-varying output**: Power follows ocean wave statistics and device dynamics
        - **Grid integration**: Modeled as controllable renewable generator
        - **Realistic profiles**: Based on physics-based WEC-Sim simulations
        
    Data Structure:
        - **Primary DataFrame**: 5-minute resolution for grid compatibility
        - **Full DataFrame**: Original simulation resolution for detailed analysis
        - **Time indexing**: Pandas datetime index for efficient time-series operations
        - **Power units**: Active power in MW, reactive power in MVAr
        - **Base power**: MVA rating for per-unit system calculations
        
    Grid Integration:
        - **PSS®E modeling**: Appears as renewable generator with time-varying output
        - **PyPSA modeling**: Integrated as generator with time-series profile
        - **Power flow studies**: Variable injection at specified bus location
        - **Stability analysis**: Dynamic renewable generation characteristics
        
    Wave Energy Modeling:
        - **Hydrodynamic response**: Power output includes wave-device interactions
        - **Control systems**: Output reflects WEC control strategies and constraints
        - **Device efficiency**: Power includes conversion losses and limitations
        - **Environmental conditions**: Output varies with wave height, period, direction
        
    Usage in WEC Farms:
        - **Building block**: Multiple devices aggregated to form utility-scale farms
        - **Homogeneous operation**: Farm devices typically share identical profiles
        - **Scalable modeling**: Linear power scaling for different farm sizes
        - **Independent objects**: Each device maintains separate data structures
        
    Performance Considerations:
        - **Memory usage**: Time-series data scales with simulation duration
        - **Data access**: Pandas DataFrame enables efficient time-based queries
        - **Copying overhead**: Each farm device maintains independent data copy
        - **Index operations**: Datetime indexing optimizes temporal lookups
        
    Notes:
        - WEC devices typically operate at unity power factor (q ≈ 0)
        - Power output can be negative during wave energy absorption phases
        - Base power rating enables consistent per-unit calculations
        - Device data derived from validated WEC-Sim hydrodynamic simulations
        - Time-series length limited by available WEC-Sim simulation duration
        
    See Also:
        WECFarm: Collection of WECDevice objects for utility-scale modeling
        WECSimRunner: Generates the simulation data used by WECDevice
        Engine: High-level interface for WEC integration in power systems
        
    References:
        WEC-Sim documentation: https://wec-sim.github.io/WEC-Sim/
        IEC 62600-2: Marine energy systems design requirements
    """
    name: str
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    dataframe_full: pd.DataFrame = field(default_factory=pd.DataFrame)
    base: Optional[float] = None  # 100 MVA probably
    bus_location: Optional[int] = None
    model: Optional[str] = None
    sim_id: Optional[int] = None

    def __repr__(self) -> str:
        """Return a formatted string representation of the WEC device configuration.
        
        Provides a hierarchical display of key device parameters for debugging,
        logging, and user information. The format shows essential device
        characteristics including identification, grid connection, and data status.
        
        Returns:
            str: Formatted multi-line string with device configuration details.
                Includes device name, model type, grid connection parameters,
                simulation metadata, base power rating, and data size in a
                tree-like structure for easy reading.
                
        Example:
            >>> device = WECDevice(
            ...     name="RM3_101_0",
            ...     model="RM3",
            ...     bus_location=14,
            ...     sim_id=101,
            ...     base=100.0,
            ...     dataframe=power_data  # 288 rows
            ... )
            >>> print(device)
            WECDevice:
            ├─ name: 'RM3_101_0'
            ├─ model: 'RM3'
            ├─ bus_location: 14
            ├─ sim_id: 101
            ├─ base: 100.0 MVA
            └─ rows: 288
            
        Display Format:
            - **Tree structure**: Uses Unicode box-drawing characters
            - **Device identification**: Name and model type in quotes
            - **Grid parameters**: Bus location and simulation ID as integers
            - **Power rating**: Base power with MVA units
            - **Data size**: Number of time-series data points
            
        Information Categories:
            - **Identity**: Device name (typically includes model and index)
            - **Type**: WEC model for hydrodynamic characteristics
            - **Grid connection**: Bus location for electrical network modeling
            - **Simulation link**: Database ID for traceability
            - **Power rating**: Base MVA for per-unit calculations
            - **Data status**: Time-series length for validation
            
        Use Cases:
            - **Interactive debugging**: Quick device configuration inspection
            - **Jupyter notebooks**: Clean display in research environments
            - **Logging output**: Structured device information for log files
            - **Data validation**: Verify device setup and data availability
            - **Farm inspection**: Review individual devices in large collections
            
        Notes:
            - Name and model shown in quotes to distinguish strings
            - Row count reflects primary dataframe length (grid integration data)
            - Missing values displayed as None for optional parameters
            - Unicode characters may not display properly in all terminals
            - Format consistent with WECFarm.__repr__() for visual coherence
            
        See Also:
            WECFarm.__repr__: Similar formatting for farm-level display
        """
        return f"""WECDevice:
    ├─ name: {self.name!r}
    ├─ model: {self.model!r}
    ├─ bus_location: {self.bus_location}
    ├─ sim_id: {self.sim_id}
    ├─ base: {"{} MVA".format(self.base)}
    └─ rows: {len(self.dataframe)}
    """