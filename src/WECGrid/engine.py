# src/wecgrid/engine.py

from datetime import datetime
from typing import List, Optional, Dict
import os
import pandas as pd
import numpy as np

from pathlib import Path

from typing import Union


from wecgrid.database.wecgrid_db import WECGridDB
from wecgrid.modelers import PSSEModeler, PyPSAModeler
from wecgrid.plot import WECGridPlotter
from wecgrid.wec import WECFarm, WECSimRunner
from wecgrid.util import WECGridTimeManager
from wecgrid.util.resources import resolve_grid_case


from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd


#TODO figure out wec-sim source "wec_sim": "C:/Users/alexb/research/WEC-Sim",

class Engine:
    """Main orchestrator for WEC-Grid simulations and cross-platform power system analysis.
    
    The Engine class serves as the central coordinator for WEC-Grid simulations, integrating
    Wave Energy Converter (WEC) farms with power system modeling across multiple software
    platforms (PSS®E and PyPSA). It manages simulation workflows, time coordination,
    database operations, and visualization.
    
    Key Capabilities:
        - **Multi-platform simulation**: Supports both PSS®E and PyPSA backends
        - **WEC farm integration**: Manages multiple WEC farms with realistic power output
        - **Time series coordination**: Handles synchronized time-series simulations
        - **Load profile generation**: Creates realistic load curves with demand variability
        - **Database integration**: Stores and retrieves simulation results
        - **Visualization interface**: Provides comprehensive plotting capabilities
        
    Attributes:
        case_file (str, optional): Path to the power system case file (.RAW format).
        case_name (str, optional): Human-readable name derived from case file.
        time (WECGridTimeManager): Time coordination and snapshot management.
        psse (PSSEModeler, optional): PSS®E simulation interface.
        pypsa (PyPSAModeler, optional): PyPSA simulation interface.
        wec_farms (List[WECFarm]): Collection of WEC farms in the simulation.
        database (WECGridDB): Database interface for simulation data storage.
        plot (WECGridPlotter): Visualization and plotting interface.
        wec_sim (WECSimRunner): WEC-Sim integration for device-level modeling.
        
    Example:
        >>> # Basic setup and simulation
        >>> engine = Engine()
        >>> engine.case("IEEE_30_bus")
        >>> engine.load(["psse", "pypsa"])
        >>> engine.apply_wec("North Farm", size=5, bus_location=14)
        >>> engine.simulate(sim_length=288, load_curve=True)
        
        >>> # Cross-platform comparison
        >>> engine.plot.comparison_suite()
        
        >>> # WEC-specific analysis
        >>> engine.plot.plot_wec_analysis("psse")
        
    Notes:
        - Requires appropriate software licenses (PSS®E) and installations
        - WEC data sourced from WEC-Sim simulations or database
        - Supports academic research and commercial power system studies
        - Designed for cross-platform validation and verification workflows
        
    See Also:
        PSSEModeler: PSS®E power system simulation interface
        PyPSAModeler: PyPSA power system simulation interface
        WECFarm: Wave energy converter farm modeling
        WECGridPlotter: Visualization and analysis tools
    """
    #TODO name it WECGridEngine? think on it

    def __init__(
        self
    ):
        """Initialize the WEC-Grid Engine with default configuration.
        
        Creates a new Engine instance with empty power system modelers, time management,
        database connection, and visualization interface. The engine is ready for case
        loading and simulation setup.
        
        Initializes:
            - Empty case configuration (no power system loaded)
            - Time manager with default settings
            - Database connection for simulation data storage
            - Plotting interface linked to this engine
            - WEC-Sim runner for device-level simulations
            - Empty collections for WEC farms
            
        Example:
            >>> engine = Engine()
            >>> print(f"Engine initialized with {len(engine.wec_farms)} WEC farms")
            Engine initialized with 0 WEC farms
            
        Notes:
            - No power system case is loaded initially
            - All modelers (PSS®E, PyPSA) are set to None until explicitly loaded
            - Database connection is established but no simulation data exists
            - Ready for case loading via engine.case() method
            
        See Also:
            case: Load a power system case file
            load: Initialize power system software backends
        """
        self.case_file: Optional[str] = None
        self.case_name: Optional[str] = None
        self.time = WECGridTimeManager() # TODO this needs more functionality
        #self.path_manager = WECGridPathManager()
        self.psse: Optional[PSSEModeler] = None
        self.pypsa: Optional[PyPSAModeler] = None
        self.wec_farms: List[WECFarm] = []
        self.database = WECGridDB()
        self.plot = WECGridPlotter(self)
        self.wec_sim: WECSimRunner = WECSimRunner(self.database)


    def case(self, case_file: str):
        """Load a power system case file for simulation.
        
        Sets the power system case file that will be used for all subsequent simulations.
        Supports both local file paths and bundled IEEE test cases. The case name is
        automatically derived from the filename for display purposes.
        
        Args:
            case_file (str): Path to power system case file. Can be:
                - Full path to local .RAW file: "/path/to/system.RAW"
                - Relative path: "data/ieee_30.RAW" 
                - Bundled case name: "IEEE_30_bus" or "IEEE_14_bus"
                - Case with extension: "IEEE_39_bus.RAW"
        
        Returns:
            None: Sets internal case_file and case_name attributes.
            
        Raises:
            FileNotFoundError: If specified case file cannot be located.
            ValueError: If case file format is not supported.
            
        Example:
            >>> # Load bundled IEEE test case
            >>> engine.case("IEEE_30_bus")
            >>> print(f"Loaded: {engine.case_name}")
            Loaded: IEEE 30 bus
            
            >>> # Load local case file
            >>> engine.case("/path/to/custom_system.RAW")
            >>> print(f"Case file: {engine.case_file}")
            
        Supported Case Formats:
            - **PSS®E RAW**: Standard power flow data format
            - **IEEE Test Systems**: Pre-configured benchmark cases
                * IEEE_14_bus: 14-bus test system
                * IEEE_24_bus: IEEE RTS-24 reliability test system
                * IEEE_30_bus: 30-bus test system
                * IEEE_39_bus: New England 39-bus system
                
        Notes:
            - Case file path is resolved using internal resource management
            - Case name formatting removes underscores and hyphens for display
            - Must be called before load() to initialize power system backends
            - Case file is validated when modelers are initialized
            
        See Also:
            load: Initialize power system software with loaded case
            resolve_grid_case: Internal case file resolution utility
        """
        path = resolve_grid_case(case_file)
        self.case_file = str(path)
        self.case_name = Path(path).stem.replace("_", " ").replace("-", " ")
            

    def load(self, software: List[str]) -> None:
        """Initialize power system simulation backends.
        
        Initializes one or more power system modeling platforms using the previously
        loaded case file. Each software backend provides independent simulation
        capabilities and can be used for cross-platform validation.
        
        Args:
            software (List[str]): List of software backends to initialize.
                Supported options:
                - "psse": PSS®E power system simulator
                - "pypsa": PyPSA open-source power system analysis
                
        Returns:
            None: Initializes internal modeler objects.
            
        Raises:
            ValueError: If no case file is loaded or unsupported software specified.
            RuntimeError: If software initialization fails (e.g., missing licenses).
            
        Example:
            >>> # Load single backend
            >>> engine.case("IEEE_30_bus")
            >>> engine.load(["psse"])
            
            >>> # Load both backends for comparison
            >>> engine.load(["psse", "pypsa"])
            >>> print(f"PSS®E loaded: {engine.psse is not None}")
            >>> print(f"PyPSA loaded: {engine.pypsa is not None}")
            PSS®E loaded: True
            PyPSA loaded: True
            
        Software Requirements:
            **PSS®E**:
                - Valid PSS®E license and installation
                - Python API properly configured
                - Compatible PSS®E version (v33+)
                
            **PyPSA**:
                - PyPSA library installed
                - Compatible with pandapower for power flow
                - No licensing requirements (open source)
                
        Backend Initialization:
            - **PSS®E**: Initializes COM interface and loads case file
            - **PyPSA**: Converts RAW file format and builds network model
            - Both backends validate case file compatibility
            - Reactive power limits adjusted for cross-platform consistency
            
        Notes:
            - Case file must be loaded first using engine.case()
            - Multiple backends enable cross-platform validation studies
            - PSS®E requires commercial license; PyPSA is open source
            - Backends are independent and can simulate separately
            - TODO: Add error handling for initialization failures
            
        See Also:
            case: Load power system case file
            PSSEModeler: PSS®E simulation interface
            PyPSAModeler: PyPSA simulation interface
        """
        if self.case_file is None:
            raise ValueError("No case file set. Use `engine.case('path/to/case.RAW')` first.")
        
        for name in software:
            name = name.lower()
            if name == "psse":
                self.psse = PSSEModeler(self)
                self.psse.init_api()
                #TODO: check if error is thrown if init fails
            elif name == "pypsa":
                self.pypsa = PyPSAModeler(self)
                self.pypsa.init_api()
                # if self.psse is not None:
                #     self.psse.adjust_reactive_lim()
                #TODO: check if error is thrown if init fails
            else:
                raise ValueError(f"Unsupported software: '{name}'. Use 'psse' or 'pypsa'.")

    def apply_wec(
        self,
        farm_name: str,
        size: int = 1,
        sim_id: int = -1,
        model: str = "RM3",
        bus_location: int = 1,
        connecting_bus: int = 1, # todo this should default to swing bus
    ) -> None:
        """Add a Wave Energy Converter (WEC) farm to the power system simulation.
        
        Creates a WEC farm object with specified parameters and integrates it into
        all loaded power system modelers. The farm includes realistic power output
        time series based on WEC-Sim device-level simulations.
        
        Args:
            farm_name (str): Human-readable name for the WEC farm.
            size (int, optional): Number of WEC devices in the farm. Defaults to 1.
            sim_id (int, optional): Database simulation ID for WEC data retrieval.
                Use -1 for synthetic/default data. Defaults to -1.
            model (str, optional): WEC device model type. Defaults to "RM3".
                Supported models:
                - "RM3": Reference Model 3 (point absorber)
            bus_location (int, optional): Power system bus for WEC connection.
                Defaults to 1.
            connecting_bus (int, optional): Connection bus for network topology.
                Defaults to 1.
                
        Returns:
            None: Creates WECFarm object and adds to internal collections.
            
        Raises:
            ValueError: If specified WEC model is not supported.
            DatabaseError: If sim_id specified but WEC data not found.
            
        Example:
            >>> # Single WEC device
            >>> engine.apply_wec("Test WEC", size=1, bus_location=14)
            
            >>> # Large offshore wind farm
            >>> engine.apply_wec(
            ...     farm_name="North Coast Farm",
            ...     size=20,
            ...     model="RM3", 
            ...     bus_location=14,
            ...     sim_id=12
            ... )
            
            >>> print(f"Total farms: {len(engine.wec_farms)}")
            Total farms: 2
            
        WEC Farm Integration:
            - **Power Output**: Time-series active power from WEC-Sim simulations
            - **Grid Connection**: Modeled as generator at specified bus location
            - **Farm Scaling**: Individual device power multiplied by farm size
            - **Database Linking**: Retrieves device data using sim_id
            
        Power System Integration:
            - Added to PSS®E as dynamic generator with time-varying output
            - Added to PyPSA as renewable generator with time series
            - Unique generator ID assigned automatically
            - Consistent modeling across both software platforms
            
        Notes:
            - Farm size scales individual device power output linearly
            - WEC data sourced from database using sim_id parameter
            - Generator ID limited by PSS®E constraints (2-character limit)
            - Maximum 9 farms supported due to PSS®E generator ID limitations
            - TODO: Address PSS®E generator ID limitation for larger studies
            - TODO: Default connecting_bus should be swing bus
            
        WEC Farm Attributes:
            Each created farm has:
            - farm_name: Human-readable identifier
            - size: Number of devices
            - bus_location: Grid connection point
            - gen_id: Unique generator identifier
            - wec_devices: Collection of device objects with power data
            
        See Also:
            WECFarm: Wave energy converter farm modeling class
            WECSimRunner: Interface to WEC-Sim device simulations
            PSSEModeler.add_wec_farm: PSS®E integration method
            PyPSAModeler.add_wec_farm: PyPSA integration method
        """
        wec_farm: WECFarm = WECFarm(
            farm_name=farm_name,
            database=self.database,
            time=self.time,
            sim_id= sim_id,
            model=model,
            bus_location=bus_location, 
            connecting_bus=connecting_bus,
            size=size,
            gen_id= len(self.wec_farms) + 1,  # Unique gen_id for each farm,
            #TODO potenital issue where PSSE is using gen_id as the gen identifer and that's limited to 2 chars. so hard cap at 9 farms in this code rn

        )
        self.wec_farms.append(wec_farm)
        
        for modeler in [self.psse, self.pypsa]:
                if modeler is not None:
                    modeler.add_wec_farm(wec_farm)


    def generate_load_curves(
            self,
            morning_peak_hour: float = 8.0,
            evening_peak_hour: float = 18.0,
            morning_sigma_h: float = 2.0,
            evening_sigma_h: float = 3.0,
            amplitude: float = 0.30,   # ±30% swing around mean
            min_multiplier: float = 0.70,  # floor/ceiling clamp
            amp_overrides: Optional[Dict[int, float]]  = None,
        ) -> pd.DataFrame:
            """Generate realistic time-varying load profiles for power system simulation.
            
            Creates bus-specific load time series based on a normalized double-peak daily
            pattern representing typical electrical demand profiles. The method scales
            base load values from the power system case using configurable peak timing
            and variability parameters.
            
            Args:
                morning_peak_hour (float, optional): Time of morning demand peak [hours].
                    Defaults to 8.0 (8:00 AM).
                evening_peak_hour (float, optional): Time of evening demand peak [hours].
                    Defaults to 18.0 (6:00 PM).
                morning_sigma_h (float, optional): Standard deviation for morning peak [hours].
                    Controls peak width. Defaults to 2.0.
                evening_sigma_h (float, optional): Standard deviation for evening peak [hours].
                    Controls peak width. Defaults to 3.0.
                amplitude (float, optional): Maximum variation amplitude around base load.
                    Value of 0.30 means ±30% variation. Defaults to 0.30.
                min_multiplier (float, optional): Minimum load multiplier (floor constraint).
                    Defaults to 0.70 (70% of base load minimum).
                amp_overrides (Dict[int, float], optional): Per-bus amplitude overrides.
                    Keys are bus numbers, values are custom amplitudes. Defaults to None.
                    
            Returns:
                pd.DataFrame: Time-indexed DataFrame with load profiles [MW].
                    - Index: Simulation time snapshots
                    - Columns: Bus numbers with non-zero loads
                    - Values: Active power demand [MW]
                    
            Raises:
                ValueError: If no power system modeler is loaded.
                
            Example:
                >>> # Standard residential/commercial profile
                >>> load_df = engine.generate_load_curves()
                >>> print(f"Load shape: {load_df.shape}")
                Load shape: (288, 15)  # 288 time steps, 15 load buses
                
                >>> # Industrial profile with late evening peak
                >>> industrial_load = engine.generate_load_curves(
                ...     morning_peak_hour=7.0,
                ...     evening_peak_hour=22.0,
                ...     amplitude=0.15  # Less variability
                ... )
                
                >>> # Custom per-bus profiles
                >>> custom_load = engine.generate_load_curves(
                ...     amp_overrides={14: 0.50, 30: 0.10}  # High/low variability buses
                ... )
                
            Load Profile Characteristics:
                - **Double-peak pattern**: Morning and evening demand peaks
                - **Gaussian shape**: Smooth transitions between demand levels
                - **Configurable timing**: Adjustable peak hours for different regions
                - **Variable width**: Different peak durations via sigma parameters
                - **Per-bus customization**: Individual amplitude scaling
                - **Realistic constraints**: Floor/ceiling limits prevent unrealistic values
                
            Temporal Behavior:
                - **Long simulations (>6h)**: Full double-peak daily profile
                - **Short simulations (<6h)**: Flat profile to avoid artificial peaks
                - **Base load scaling**: Original case loads used as reference values
                - **Time-synchronized**: Matches engine.time.snapshots exactly
                
            Mathematical Model:
                For each time t and bus b:
                ```
                shape(t) = Σ exp(-0.5 * ((hour(t) - peak_hour) / sigma)²)
                normalized(t) = (shape(t) - mean(shape)) / std(shape)
                load(b,t) = base_load(b) * (1 + amplitude(b) * normalized(t))
                load(b,t) = clip(load(b,t), min_multiplier * base_load(b), 2-min_multiplier * base_load(b))
                ```
                
            Data Sources:
                - **PSS®E**: Extracts base loads from psse.grid.load DataFrame
                - **PyPSA**: Extracts base loads from pypsa.network.loads, aggregated by bus
                - **Automatic selection**: Uses available modeler (PSS®E preferred)
                
            Notes:
                - Zero-load buses are excluded from output DataFrame
                - Time series length matches engine time manager snapshots
                - Amplitude overrides enable modeling different customer classes
                - Min/max multipliers prevent unrealistic demand swings
                - Suitable for daily, multi-day, or seasonal studies
                
            Use Cases:
                - **Load variability studies**: Impact of demand fluctuations
                - **WEC integration analysis**: Renewable vs. demand correlation
                - **Grid stability assessment**: Dynamic load effects
                - **Economic dispatch**: Time-varying load costs
                
            See Also:
                simulate: Use generated load curves in time-series simulation
                WECGridTimeManager: Time coordination and snapshot management
                PSSEModeler.simulate: PSS®E simulation with load curves
                PyPSAModeler.simulate: PyPSA simulation with load curves
            """

            if self.psse is None and self.pypsa is None:
                raise ValueError("No power system modeler loaded. Use `engine.load(...)` first.")
            
                        # --- Use PSSE or PyPSA Grid state to get base load ---
            if self.psse is not None:
                base_load = (
                    self.psse.grid.load[["bus", "p"]]
                    .drop_duplicates("bus")
                    .set_index("bus")["p"]
                )
            elif self.pypsa is not None:
                base_load = (
                    self.pypsa.network.loads[["bus", "p"]]
                    .groupby("bus")["p"]
                    .sum()
                )
            else:
                raise ValueError("No valid base load could be extracted from modelers.")

                
            snaps = pd.to_datetime(self.time.snapshots)
            prof = pd.DataFrame(index=snaps)

            # make sure this is a plain ndarray, not a Float64Index
            hours = (snaps.hour.values
                    + snaps.minute.values/60.0
                    + snaps.second.values/3600.0)

            dur_sec = 0 if len(snaps) < 2 else (snaps.max() - snaps.min()).total_seconds()

            if dur_sec < 6*3600:
                z = np.zeros_like(hours, dtype=float)
            else:
                def g(h, mu, sig):
                    h = np.asarray(h, dtype=float)      # <-- belt-and-suspenders
                    return np.exp(-0.5 * ((h - mu)/sig)**2)

                s = g(hours, morning_peak_hour, morning_sigma_h) + g(hours, evening_peak_hour, evening_sigma_h)
                s = np.asarray(s, dtype=float)
                z = (s - s.mean()) / (s.std() + 1e-12)  # or: z = (s - np.mean(s)) / (np.std(s) + 1e-12)

            amp_overrides = {} if amp_overrides is None else {int(k): float(v) for k, v in amp_overrides.items()}

            for bus, p_base in base_load.items():
                if p_base <= 0: 
                    continue
                a = amp_overrides.get(int(bus), amplitude)   # per-bus amplitude
                shape_bus = 1.0 + a * z
                shape_bus = np.clip(shape_bus, min_multiplier, 2.0 - min_multiplier)
                prof[int(bus)] = p_base * shape_bus

            prof.index.name = "time"
            return prof

    # def generate_load_curves(self) -> pd.DataFrame:
    #     """
    #     Generate synthetic load profiles using a normalized double-peak shape.
    #     Returns a DataFrame indexed by time, with one column per bus.
    #     """
    #     #TODO the double peaks should be time dependedent, need to avoid applying a double peak to a  2hour sim 

    #     if self.psse is None and self.pypsa is None:
    #         raise ValueError("No power system modeler loaded. Use `engine.load(...)` first.")

    #     # --- Use PSSE or PyPSA Grid state to get base load ---
    #     if self.psse is not None:
    #         base_load = (
    #             self.psse.grid.load[["bus", "p"]]
    #             .drop_duplicates("bus")
    #             .set_index("bus")["p"]
    #         )
    #     elif self.pypsa is not None:
    #         base_load = (
    #             self.pypsa.network.loads[["bus", "p"]]
    #             .groupby("bus")["p"]
    #             .sum()
    #         )
    #     else:
    #         raise ValueError("No valid base load could be extracted from modelers.")

    #     # --- Create time-dependent shape (double peak) ---
    #     times = pd.to_datetime(self.time.snapshots)
    #     hours = times.hour + times.minute / 60.0

    #     shape = 0.5 + 0.5 * (
    #         np.exp(-0.5 * ((hours - 8) / 2) ** 2) +
    #         np.exp(-0.5 * ((hours - 18) / 2) ** 2)
    #     )

    #     # --- Generate time-series DataFrame ---
    #     profile = pd.DataFrame(index=self.time.snapshots)

    #     for bus, base in base_load.items():
    #         if base > 0:
    #             profile[bus] = base * shape

    #     return profile

    def simulate(
        self,
        sim_length: Optional[int] = None,
        load_curve: bool = False,
        plot: bool = True
    ) -> None:
        """Execute time-series power system simulation across loaded backends.
        
        Runs coordinated simulations using all initialized power system modelers
        (PSS®E and/or PyPSA) with optional load variability and WEC farm integration.
        Automatically manages simulation length based on available WEC data and
        user specifications.
        
        Args:
            sim_length (int, optional): Number of simulation time steps to execute.
                If None, uses full available data length. If WEC farms are present,
                length is constrained by WEC data availability. Defaults to None.
            load_curve (bool, optional): Enable time-varying load profiles.
                If True, generates realistic double-peak load curves. If False,
                uses static loads from case file. Defaults to False.
            plot (bool, optional): Enable automatic plotting after simulation.
                Currently reserved for future implementation. Defaults to True.
                
        Returns:
            None: Simulation results stored in modeler objects and accessible
                via engine.psse.grid and engine.pypsa.grid.
                
        Raises:
            ValueError: If no power system modelers are loaded.
            RuntimeError: If simulation fails in any backend.
            
        Example:
            >>> # Basic simulation with static loads
            >>> engine.simulate(sim_length=144)  # 12 hours at 5-min intervals
            
            >>> # Full day simulation with load variability
            >>> engine.simulate(sim_length=288, load_curve=True)
            
            >>> # WEC-constrained simulation (auto-length detection)
            >>> engine.apply_wec("Test Farm", size=5, bus_location=14)
            >>> engine.simulate(load_curve=True)  # Length from WEC data
            
        Simulation Coordination:
            - **Time synchronization**: All backends use identical time snapshots
            - **Load profiles**: Optional time-varying demand curves
            - **WEC integration**: Realistic renewable power injection
            - **Cross-platform**: Simultaneous PSS®E and PyPSA execution
            
        Length Management:
            **Without WEC farms**:
                - Uses sim_length if specified
                - Otherwise uses engine.time default length
                
            **With WEC farms**:
                - Constrained by WEC data availability
                - Uses min(sim_length, available_wec_length)
                - Warns if requested length exceeds WEC data
                - Auto-updates time manager if length changes
                
        Load Curve Integration:
            When load_curve=True:
            - Generates double-peak daily profiles using generate_load_curves()
            - Default amplitude: 10% (±10% around base load)
            - Applied to all buses with non-zero loads
            - Replaces static case file load values
            
        Backend Execution:
            - **PSS®E**: Dynamic simulation with time-series data injection
            - **PyPSA**: Optimal power flow across all time snapshots
            - **Independent operation**: Each backend simulates separately
            - **Consistent inputs**: Same load curves and WEC data
            
        Data Storage:
            Simulation results accessible via:
            - `engine.psse.grid`: PSS®E time-series results
            - `engine.pypsa.grid`: PyPSA time-series results
            - Database storage handled by individual modelers
            
        Notes:
            - WEC data length constraint prevents simulation errors
            - Different WEC farms with varying data lengths may cause issues
            - Load curve amplitude reduced (10%) for realistic daily variation
            - Plot parameter reserved for future automatic visualization
            - TODO: Address multi-farm data length inconsistencies
            
        Performance Considerations:
            - PSS®E simulations scale with network size and time steps
            - PyPSA simulations benefit from optimization solver selection
            - Memory usage increases with longer time series
            - Parallel execution not currently supported
            
        See Also:
            generate_load_curves: Time-varying load profile generation
            PSSEModeler.simulate: PSS®E backend simulation method
            PyPSAModeler.simulate: PyPSA backend simulation method
            WECGridTimeManager: Time coordination and snapshot management
        """

        # show that if different farms have different wec durations this logic fails
        if self.wec_farms:
            available_len = len(self.wec_farms[0].wec_devices[0].dataframe)

            if sim_length is not None:
                if sim_length > available_len:
                    print(f"[WARNING] Requested sim_length={sim_length} exceeds "
                        f"WEC data length={available_len}. Truncating to {available_len}.")
                final_len = min(sim_length, available_len)
            else:
                final_len = available_len

            if final_len != self.time.snapshots.shape[0]:
                self.time.update(sim_length=final_len)

        else:
            # No WEC farm — just update if sim_length is given
            if sim_length is not None:
                self.time.update(sim_length=sim_length)

        load_curve_df = self.generate_load_curves(amplitude=0.10) if load_curve else None

        for modeler in [self.psse, self.pypsa]:
            if modeler is not None:
                modeler.simulate(load_curve=load_curve_df)
                # todo if plot then plot


# # src/wecgrid/engine.py

# """
# WEC-Grid Engine module
# Author: Alexander Barajas-Ritchie
# Email: barajale@oregonstate.edu
# """

# # Standard library
# import os
# import sys
# import time
# import json
# from datetime import datetime, timezone, timedelta
# from typing import List, Union

# # Third-party
# import pandas as pd
# import numpy as np
# import sqlite3
# import pypsa
# import pypower.api as pypower
# import matlab.engine
# import matplotlib.pyplot as plt

# # Local imports
# from wecgrid.database import wecgrid_db
# from wecgrid.modelers import PSSEModeler, PyPSAModeler
# from wecgrid.plot import WECGridPlotter
# from wecgrid.wec import Farm


# class Engine:
#     """
#     Main class for coordinating between PSSE and PyPSA functionality and managing WEC devices.

#     Attributes:
#         case (str): Path to the case file.
#         psseObj (PSSEWrapper): Instance of the PSSE wrapper class.
#         pypsaObj (PyPSAWrapper): Instance of the PyPSA wrapper class.
#         wecObj_list (list): List of WEC objects.
#     """

#     def __init__(self, case: str):
#         """
#         Initializes the WecGrid class with the given case file.

#         Args:
#             case (str): Path to the case file.
#         """
#         self.case_file = case  # TODO: need to verify file exist
#         self.case_file_name = os.path.basename(case)
#         self.psse = None
#         self.pypsa = None
#         self.start_time = datetime(1997, 11, 3, 0, 0, 0)
#         self.sim_length = 288 # 5 min intervals
#         self.snapshots = pd.date_range(
#                 start=self.start_time,
#                 periods= self.sim_length , # 288 5-minute intervals in a day
#                 freq="5T",  # 5-minute intervals
#             )
#         self.wecObj_list = []
#         self.wec_buses = []
#         self.software = [] 
#         self.generator_compare = None
#         self.bus_compare = None
#         self.viz = WECGridVisualizer(self)
        
#         self.initialize_simulation_db(DB_PATH)

#     def initialize_simulation_db(self, path):
#         #TODO: need to move this to DB wrapper handler 
#         #TODO: needs to check if db exists, if not create it with tables, if there, should check if tables exist,
#         #TODO: need a function to print a report of the contents of the database
        
#         with sqlite3.connect(path) as conn:
#             cursor = conn.cursor()

#             cursor.execute("""
#             CREATE TABLE IF NOT EXISTS sim_runs (
#                 sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 sim_name TEXT NOT NULL,
#                 timestamp TEXT NOT NULL,
#                 notes TEXT
#             )
#             """)

#             cursor.execute("""
#             CREATE TABLE IF NOT EXISTS bus_timeseries (
#                 sim_id INTEGER,
#                 timestamp TEXT,
#                 bus_id TEXT,
#                 p_mw REAL,
#                 v_pu REAL,
#                 source TEXT,
#                 FOREIGN KEY(sim_id) REFERENCES sim_runs(sim_id)
#             )
#             """)

#             cursor.execute("""
#             CREATE TABLE IF NOT EXISTS gen_timeseries (
#                 sim_id INTEGER,
#                 timestamp TEXT,
#                 gen_id TEXT,
#                 p_mw REAL,
#                 source TEXT,
#                 FOREIGN KEY(sim_id) REFERENCES sim_runs(sim_id)
#             )
#             """)

#             conn.commit()
        
#     def load(self, software):
#         """
#         Enables one or more supported power system software tools.

#         Args:
#             software (str or list of str): Name(s) of supported software to initialize.
#                                         Options: "psse", "pypsa"
#         """
#         if isinstance(software, str):
#             software = [software]

#         for name in software:
#             name = name.lower()
#             if name == "psse":
#                 self.psse = PSSEInterface(self.case_file, self)
#                 self.psse.init_api()
#             elif name == "pypsa":
#                 self.pypsa = PYPSAInterface(self.case_file, self)
#                 self.pypsa.init_api()
#                 if self.psse is not None:
#                     self.psse.adjust_reactive_lim()
#             else:
#                 raise ValueError(f"Unsupported software: '{name}'. Use 'psse' or 'pypsa'.")

#     def apply_wecs(self, sim_id=None, model="RM3", farm_size=8, ibus=None, jbus=1, mbase=0.01, config=None):
#         #TODO: need to confirm i and j bus are correct orientation
#         """
#         Creates a WEC device and adds it to both PSSE and PyPSA models.

#         Args:
#             ID (int): Identifier for the WEC device.
#             model (str): Model type of the WEC device.
#             from_bus (int): The bus number from which the WEC device is connected.
#             to_bus (int): The bus number to which the WEC device is connected.
#         """
#         if config is None:
#             config = {
#                 "simLength": (self.sim_length * 5 * 60), # Simulation length in seconds
#                 "Tsample": 300,  # Sampling time of 5 minutes
#                 "waveHeight": 2.5, # Wave height in meters
#                 "wavePeriod": 8, # Wave period in seconds
#             }
#         else:
#             self.sim_length = config["simLength"] / 300
#             self.snapshots = pd.date_range(
#                 start=self.start_time,
#                 periods= self.sim_length , # 288 5-minute intervals in a day
#                 freq="5T",  # 5-minute intervals
#             )
#         self.wec_buses.append(ibus)
#         for i in range(farm_size):
#             self.wecObj_list.append(
#                 device.WEC(
#                     engine=self,
#                     sim_id=sim_id,
#                     model=model,
#                     bus_location=ibus,
#                     MBASE=mbase,
#                     config=config  
#                 )
#             )
#         if self.pypsa is not None:
#             #TODO: this is not returning false if broken
#             if self.pypsa.add_wec(model, ibus, jbus):
#                 print("WEC components added to PyPSA network.")
#             else:
#                 print("Failed to add WEC to PyPSA network.")
            
#         if self.psse is not None:
#             if self.psse.add_wec(model, ibus, jbus):
#                 print("WEC components added to PSS®E network.")
#             else:
#                 print("Failed to add WEC to PSS®E network.")

#     def generate_load_profiles(self):
#         """
#         Create a unified double-peaking load curve profile, then split it for pyPSA and PSS®E.
#         Stores results in self.load_profiles_pypsa and self.load_profiles_psse
#         """
#         # 1. Extract base loads for pyPSA (load ID → MW)
#         pypsa_base = self.pypsa.network.loads[["bus", "p_set"]].copy()

#         # 2. Extract base loads for PSS®E (bus number → MW)
#         psse_base = (
#             self.psse.loads_dataframe[["BUS_NUMBER", "P_MW"]]
#             .drop_duplicates("BUS_NUMBER")
#             .set_index("BUS_NUMBER")["P_MW"]
#         )

#         # 3. Create the normalized load shape over the day
#         times = pd.to_datetime(self.snapshots)
#         hours = times.hour + times.minute / 60.0

#         def double_peak(hour):
#             return 0.5 + 0.5 * (
#                 np.exp(-0.5 * ((hour - 8) / 2) ** 2) +
#                 np.exp(-0.5 * ((hour - 18) / 2) ** 2)
#             )

#         shape = double_peak(hours)

#         # 4. Create time-indexed DataFrames
#         df_pypsa = pd.DataFrame(index=self.snapshots)
#         df_psse = pd.DataFrame(index=self.snapshots)

#         # 5. Apply curve to pyPSA loads (column = load ID)
#         for load_id, row in pypsa_base.iterrows():
#             base = row["p_set"]
#             if base > 0:
#                 df_pypsa[load_id] = base * shape

#         # 6. Apply curve to PSS®E loads (column = bus number)
#         for bus, base in psse_base.items():
#             if base > 0:
#                 df_psse[bus] = base * shape

#         # 7. Store
#         self.pypsa.load_profiles = df_pypsa
#         self.psse.load_profiles = df_psse
        
#     def simulate(self, load_curve=True, plot=True):
#         """
#         Simulates the WEC devices and updates the PSSE and PyPSA models.

#         Args:
#             load_curve (bool): If True, simulates the load curve.
#         """
#         if self.psse is not None:
#             print("Simulating on PSS®E...")
#             start_time = time.time()
#             if self.psse.simulate(load_curve=load_curve, plot=plot):
#                 print("PSS®E simulation complete in {} seconds. \n".format(time.time() - start_time))
#             else:
#                 print("PSS®E simulation failed. \n")
            
#         if self.pypsa is not None:
#             print("Simulating on PyPSA...")
#             start_time = time.time()
#             if self.pypsa.simulate(load_curve=load_curve, plot=plot):
#                 print("PyPSA simulation complete in {} seconds. \n".format(time.time() - start_time))
#             else:
#                 print("PyPSA simulation failed. \n")
        
#         #self.compare_results()

#     def compare_results(self, plot=True):
#         if self.psse is not None and self.pypsa is not None and plot:
#             self.viz.plot_comparison()

#         def compute_rmse_corr(psse_df, pypsa_df, label_prefix=None, convert_cols=False):
#             if convert_cols:
#                 psse_df.columns = psse_df.columns.map(str)
#             common = sorted(set(psse_df.columns).intersection(set(pypsa_df.columns)), key=str)

#             rows = []
#             for key in common:
#                 psse_series = psse_df[key]
#                 pypsa_series = pypsa_df[key]
#                 rmse = np.sqrt(((psse_series - pypsa_series) ** 2).mean())

#                 # Only compute correlation if there is variance
#                 if psse_series.std() == 0 or pypsa_series.std() == 0:
#                     corr = np.nan
#                 else:
#                     corr = psse_series.corr(pypsa_series)

#                 rows.append({
#                     "ID": key,
#                     "Parameter": label_prefix,
#                     "RMSE": rmse,
#                 })
#             return pd.DataFrame(rows)

#         # Compute generator comparison
#         gen_df = compute_rmse_corr(
#             self.psse.generator_dataframe_t.p.copy(),
#             self.pypsa.network.generators_t.p.copy(),
#             label_prefix="P"
#         ).rename(columns={"ID": "Generator"})[["Generator", "Parameter", "RMSE"]]

#         # Compute bus P and V comparison
#         bus_p_df = compute_rmse_corr(
#             self.psse.bus_dataframe_t.p.copy(),
#             self.pypsa.network.buses_t.p.copy(),
#             label_prefix="P",
#             convert_cols=True
#         )

#         bus_v_df = compute_rmse_corr(
#             self.psse.bus_dataframe_t.v_mag_pu.copy(),
#             self.pypsa.network.buses_t.v_mag_pu.copy(),
#             label_prefix="V_mag",
#             convert_cols=True
#         )

#         bus_df = pd.concat([bus_p_df, bus_v_df]) \
#                 .rename(columns={"ID": "Bus"}) \
#                 [["Bus", "Parameter", "RMSE"]]

#         # Save to attributes for inspection in notebook
#         self.generator_compare = gen_df
#         self.bus_compare = bus_df

#     def save_simulation(self, sim_name="Unnamed Run", notes=""):
#         timestamp = datetime.now().isoformat()
#         with sqlite3.connect(DB_PATH) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO sim_runs (sim_name, timestamp, notes)
#                 VALUES (?, ?, ?)
#             """, (sim_name, timestamp, notes))
#             sim_id = cursor.lastrowid

#         if self.psse is not None:
#             self.save_psse_run(sim_id)
#         if self.pypsa is not None:
#             self.save_pypsa_run(sim_id)

#     def save_psse_run(self, sim_id):
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         psse_bus = self.psse.bus_dataframe_t
#         psse_gen = self.psse.generator_dataframe_t

#         for timestamp in psse_bus.p.index:
#             for bus in psse_bus.p.columns:
#                 cursor.execute("""
#                     INSERT INTO bus_timeseries (sim_id, timestamp, bus_id, p_mw, v_pu, source)
#                     VALUES (?, ?, ?, ?, ?, ?)
#                 """, (sim_id, timestamp.isoformat(), str(bus),
#                     psse_bus.p.at[timestamp, bus],
#                     psse_bus.v_mag_pu.at[timestamp, bus],
#                     "psse"))

#             for gen in psse_gen.p.columns:
#                 cursor.execute("""
#                     INSERT INTO gen_timeseries (sim_id, timestamp, gen_id, p_mw, source)
#                     VALUES (?, ?, ?, ?, ?)
#                 """, (sim_id, timestamp.isoformat(), gen,
#                     psse_gen.p.at[timestamp, gen],
#                     "psse"))

#         conn.commit()
#         conn.close()

#     def save_pypsa_run(self, sim_id):
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()

#         pypsa_bus = self.pypsa.network.buses_t
#         pypsa_gen = self.pypsa.network.generators_t

#         for timestamp in pypsa_bus.p.index:
#             for bus in pypsa_bus.p.columns:
#                 cursor.execute("""
#                     INSERT INTO bus_timeseries (sim_id, timestamp, bus_id, p_mw, v_pu, source)
#                     VALUES (?, ?, ?, ?, ?, ?)
#                 """, (sim_id, timestamp.isoformat(), str(bus),
#                     pypsa_bus.p.at[timestamp, bus],
#                     pypsa_bus.v_mag_pu.at[timestamp, bus],
#                     "pypsa"))

#             for gen in pypsa_gen.p.columns:
#                 cursor.execute("""
#                     INSERT INTO gen_timeseries (sim_id, timestamp, gen_id, p_mw, source)
#                     VALUES (?, ?, ?, ?, ?)
#                 """, (sim_id, timestamp.isoformat(), gen,
#                     pypsa_gen.p.at[timestamp, gen],
#                     "pypsa"))

#         conn.commit()
#         conn.close()

#     def saved_runs(self):
#         """
#         Lists all simulation runs stored in the database.
#         """
#         with sqlite3.connect(DB_PATH) as conn:
#             df = pd.read_sql("SELECT * FROM sim_runs ORDER BY sim_id DESC", conn)
#         return df
    
#     def pull_sim(self, sim_id=None):
#         """
#         Loads the results of a previous simulation run.

#         Args:
#             sim_id (int, optional): If not provided, loads the most recent run.

#         Returns:
#             dict: A dictionary containing DataFrames:
#                 {
#                     "psse_gen": ..., "pypsa_gen": ...,
#                     "psse_bus": ..., "pypsa_bus": ...
#                 }
#         """
#         with sqlite3.connect(DB_PATH) as conn:
#             cursor = conn.cursor()

#             if sim_id is None:
#                 cursor.execute("SELECT MAX(sim_id) FROM sim_runs")
#                 sim_id = cursor.fetchone()[0]
#                 if sim_id is None:
#                     raise ValueError("No simulation runs found in the database.")

#             gen_df = pd.read_sql(
#                 "SELECT * FROM gen_timeseries WHERE sim_id = ?", conn, params=(sim_id,)
#             )
#             bus_df = pd.read_sql(
#                 "SELECT * FROM bus_timeseries WHERE sim_id = ?", conn, params=(sim_id,)
#             )

#         # Split by source
#         gen_psse = gen_df[gen_df["source"] == "psse"].drop(columns=["sim_id", "source"])
#         gen_pypsa = gen_df[gen_df["source"] == "pypsa"].drop(columns=["sim_id", "source"])
#         bus_psse = bus_df[bus_df["source"] == "psse"].drop(columns=["sim_id", "source"])
#         bus_pypsa = bus_df[bus_df["source"] == "pypsa"].drop(columns=["sim_id", "source"])

#         # Timestamp to index
#         for df in [gen_psse, gen_pypsa, bus_psse, bus_pypsa]:
#             df["timestamp"] = pd.to_datetime(df["timestamp"])
#             df.set_index("timestamp", inplace=True)

#         # Pivot helper (flexible to dropped columns)
#         def pivot(df, id_col):
#             return {
#                 col: df.pivot_table(index="timestamp", columns=id_col, values=col)
#                 for col in df.columns
#                 if col != id_col
#             }

#         return {
#             "psse_gen": pivot(gen_psse, "gen_id"),
#             "pypsa_gen": pivot(gen_pypsa, "gen_id"),
#             "psse_bus": pivot(bus_psse, "bus_id"),
#             "pypsa_bus": pivot(bus_pypsa, "bus_id"),
#         }
    

# #TODO: need to update the software to run without wec case 
# #TODO: need a function to tell me all the wec-sim runs i have in my db
# #TODO: need a function to store my simulation results in the db
# #TODO: need a function to pull my simulation results from the db for analysis? 
 
            
            
            
