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
        >>> engine.apply_wec("North Farm", size=5, bus_location=14, model="RM3")
        >>> engine.simulate(sim_length=288, load_curve=True)
        
        >>> # Cross-platform comparison
        >>> engine.plot.comparison_suite()
        
        >>> # WEC-specific analysis
        >>> engine.plot.plot_wec_analysis("psse")
        
    Notes:
        - Requires appropriate software licenses (PSS®E) and installations
        - WEC data sourced from WEC-Sim simulations or database. MATLAB required for WEC-SIM simulations
        - Supports academic research and commercial power system studies
        - Designed for cross-platform validation and verification workflows
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
            
        Notes:
            - No power system case is loaded initially
            - All modelers (PSS®E, PyPSA) are set to None until explicitly loaded
            - Database connection is established but no data is pulled
            - Ready for case loading via engine.case() method
            
        """
        self.case_file: Optional[str] = None
        self.case_name: Optional[str] = None
        self.time = WECGridTimeManager() # TODO this needs more functionality
        self.psse: Optional[PSSEModeler] = None
        self.pypsa: Optional[PyPSAModeler] = None
        self.wec_farms: List[WECFarm] = []
        self.database = WECGridDB()
        self.plot = WECGridPlotter(self)
        self.wec_sim: WECSimRunner = WECSimRunner(self.database)


    def case(self, case_file: str):
        """Load a power system case file for simulation.
        
        Sets the power system case file that will be used for all subsequent simulations 
        and Power System modlers. Supports both local file paths and bundled IEEE test cases. 
        The case name is automatically derived from the filename for display purposes.
        
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
               
        Notes:
            - Case file must be loaded first using engine.case()
            - Multiple backends enable cross-platform validation studies
            - PSS®E requires commercial license; PyPSA is open source
            - Backends are independent and can simulate separately
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
        sim_id: int = 1,
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
                Use 1 for default data.
            model (str, optional): WEC device model type. Defaults to "RM3".
                Supported models:
                - "RM3": Reference Model 3 (point absorber)
            bus_location (int, optional): Power system bus for WEC connection.
                Defaults to 1.
            connecting_bus (int, optional): Connection bus for network topology.
                Defaults to 1. Typically the swing bus.
                
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
            
        Notes:
            - Farm size scales individual device power output linearly
            - WEC data sourced from database using sim_id parameter
            - Generator ID limited by PSS®E constraints (2-character limit)
            - Maximum 9 farms supported due to PSS®E generator ID limitations
            - TODO: Address PSS®E generator ID limitation for larger studies
            - TODO: Default connecting_bus should be swing bus
            
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
            
        Notes:
            - WEC data length constraint prevents simulation errors
            - Different WEC farms with varying data lengths may cause issues
            - Load curve amplitude reduced (10%) for realistic daily variation
            - Plot parameter reserved for future automatic visualization
            - TODO: Address multi-farm data length inconsistencies

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
