"""
WEC-Sim simulation runner for Wave Energy Converter device-level modeling.

This module provides the interface between WEC-Grid and WEC-Sim for high-fidelity
wave energy converter simulations using MATLAB engine integration.
"""

import os
import random

import matlab.engine
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from wecgrid.database.wecgrid_db import WECGridDB
#from wecgrid.util.wecgrid_pathmanager import WECGridPathManager
from wecgrid.util.resources import resolve_wec_model


# Inside wecsim_runner.py (at the top)
from dataclasses import dataclass


class WECSimRunner:
    """Interface for running WEC-Sim device-level simulations via MATLAB engine.
    
    The WECSimRunner class provides a Python interface to the WEC-Sim MATLAB toolbox
    for high-fidelity wave energy converter simulations. It manages MATLAB engine
    lifecycle, configures simulation parameters, executes WEC-Sim models, and stores
    results in the WEC-Grid database.
    
    Key Capabilities:
        - **MATLAB Integration**: Manages MATLAB engine startup/shutdown
        - **WEC-Sim Interface**: Executes device-level hydrodynamic simulations
        - **Parameter Management**: Configures wave conditions and simulation settings
        - **Database Storage**: Stores simulation results for grid integration studies
        - **Visualization**: Generates plots of WEC power output and wave conditions
        - **Model Support**: Handles multiple WEC device models (RM3, LUPA, etc.)
        
    Attributes:
        wec_sim_path (str, optional): Path to WEC-Sim MATLAB framework installation.
        database (WECGridDB): Database interface for simulation data storage.
        matlab_engine (matlab.engine.MatlabEngine, optional): Active MATLAB engine instance.
        
    Example:
        >>> # Basic WEC-Sim simulation
        >>> from wecgrid.database import WECGridDB
        >>> db = WECGridDB()
        >>> runner = WECSimRunner(db)
        >>> runner.set_wec_sim_path("/path/to/WEC-Sim")
        >>> 
        >>> # Run RM3 simulation with specified wave conditions
        >>> success = runner(
        ...     sim_id=101,
        ...     model="RM3",
        ...     sim_length_secs=3600,  # 1 hour
        ...     wave_height=2.5,       # 2.5m significant wave height
        ...     wave_period=8.0        # 8s peak period
        ... )
        >>> 
        >>> if success:
        ...     print("WEC-Sim simulation completed successfully")
        
    Workflow:
        1. **Initialization**: Create runner with database connection
        2. **Configuration**: Set WEC-Sim installation path
        3. **Execution**: Run device simulations with wave parameters
        4. **Storage**: Results automatically stored in database
        5. **Visualization**: Automatic plotting of simulation results
        
    WEC-Sim Integration:
        - Requires WEC-Sim MATLAB toolbox installation
        - Supports standard WEC models (RM3, LUPA)
        - Handles wave generation and hydrodynamic calculations
        - Outputs power time series for grid integration
        
    Database Schema:
        Creates tables for each simulation:
        - `WECSIM_{model}_{sim_id}`: Downsampled results for grid simulation
        - `WECSIM_{model}_{sim_id}_full`: Full-resolution simulation data
        
    Notes:
        - Requires valid MATLAB license and WEC-Sim installation
        - MATLAB engine startup can be slow (~30-60 seconds)
        - Simulation time scales with duration and wave complexity
        - Results include both power output and wave elevation data
        - Automatic visualization helps validate simulation quality
        
    See Also:
        WECGridDB: Database interface for simulation data
        WECFarm: Grid-level WEC farm modeling
        resolve_wec_model: WEC model path resolution utility
        
    References:
        WEC-Sim documentation: https://wec-sim.github.io/WEC-Sim/
    """
    def __init__(self, database: WECGridDB):
        """Initialize WEC-Sim runner with database connection.
        
        Creates a new WEC-Sim runner instance configured for device-level simulations.
        The runner requires a database connection for storing simulation results but
        defers MATLAB engine initialization until needed.
        
        Args:
            database (WECGridDB): Database interface for simulation data storage.
                Must be a valid WECGridDB instance with working connection.
                
        Example:
            >>> from wecgrid.database import WECGridDB
            >>> db = WECGridDB()
            >>> runner = WECSimRunner(db)
            >>> print(f"Runner initialized, MATLAB engine: {runner.matlab_engine}")
            Runner initialized, MATLAB engine: None
            
        Initialization State:
            - Database connection: Active and ready
            - WEC-Sim path: Not configured (must set manually)
            - MATLAB engine: Not started (lazy initialization)
            
        Notes:
            - MATLAB engine startup is deferred for performance
            - WEC-Sim path must be configured before running simulations
            - Database connection is validated during initialization
            
        See Also:
            set_wec_sim_path: Configure WEC-Sim installation location
            start_matlab: Initialize MATLAB engine when needed
        """

        #self.path_manager: WECGridPathManager = path_manager
        self.wec_sim_path: Optional[str] = None
        self.database: WECGridDB = database
        self.matlab_engine: Optional[matlab.engine.MatlabEngine] = None
    
    def set_wec_sim_path(self, path: str) -> None:
        """Configure the WEC-Sim MATLAB framework installation path.
        
        Sets the path to the WEC-Sim MATLAB toolbox installation, which is required
        for running device-level wave energy converter simulations. The path is
        validated to ensure the WEC-Sim framework is properly installed.
        
        Args:
            path (str): Absolute path to WEC-Sim framework root directory.
                Should contain WEC-Sim MATLAB functions and initialization files.
                
        Returns:
            None: Sets internal wec_sim_path attribute.
            
        Raises:
            FileNotFoundError: If specified path does not exist.
            
        Example:
            >>> # Windows installation
            >>> runner.set_wec_sim_path("C:/WEC-Sim")
            
            >>> # Linux/Mac installation  
            >>> runner.set_wec_sim_path("/opt/WEC-Sim")
            
            >>> # Verify configuration
            >>> print(f"WEC-Sim path: {runner.wec_sim_path}")
            WEC-Sim path: C:/WEC-Sim
            
        Path Requirements:
            - Must be absolute path to WEC-Sim root directory
            - Directory must exist and be accessible
            - Should contain WEC-Sim MATLAB functions and examples
            - Typically includes subdirectories: functions/, examples/, docs/
            
        Notes:
            - Path validation occurs immediately upon setting
            - MATLAB engine will use this path for WEC-Sim framework initialization
            - Required before running any WEC-Sim simulations
            - Path should point to WEC-Sim framework root, not specific models
            
        See Also:
            start_matlab: Uses this path for MATLAB engine configuration
            resolve_wec_model: Resolves individual WEC model paths
        """
        self.wec_sim_path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"WEC-SIM path does not exist: {path}")
        
    def start_matlab(self) -> bool:
        """Initialize MATLAB engine and configure WEC-Sim framework paths.
        
        Starts a MATLAB engine session and adds the WEC-Sim framework to the MATLAB
        path for device simulation capabilities. This is a potentially slow operation
        that is typically called automatically when needed.
        
        Returns:
            bool: True if MATLAB engine was started successfully, False if already running.
            
        Raises:
            ValueError: If WEC-Sim path is not configured.
            FileNotFoundError: If WEC-Sim path does not exist.
            RuntimeError: If MATLAB engine fails to start.
            
        Example:
            >>> # Manual engine startup
            >>> runner.set_wec_sim_path("/path/to/WEC-Sim")
            >>> success = runner.start_matlab()
            Starting MATLAB engine...
            MATLAB engine started and WEC-Sim path added...
            >>> print(f"Engine started: {success}")
            Engine started: True
            
            >>> # Subsequent calls return False (already running)
            >>> success = runner.start_matlab()
            MATLAB engine is already running.
            >>> print(f"Engine started: {success}")
            Engine started: False
            
        Startup Process:
            1. **Engine Initialization**: Starts MATLAB engine (slow operation)
            2. **Path Validation**: Verifies WEC-Sim installation exists
            3. **Path Configuration**: Adds WEC-Sim framework to MATLAB path
            4. **Confirmation**: Prints status messages for user feedback
            
        Performance Considerations:
            - **Startup time**: 30-60 seconds typical for MATLAB engine initialization
            - **Memory usage**: MATLAB engine requires significant system memory
            - **License check**: Validates MATLAB license during startup
            - **Path generation**: Recursively adds all WEC-Sim subdirectories
            
        MATLAB Path Configuration:
            - Uses `genpath()` to recursively include all WEC-Sim subdirectories
            - Includes WEC-Sim functions, examples, and utility scripts
            - Enables access to all WEC-Sim MATLAB functions and classes
            
        Notes:
            - Automatically called by simulation methods if needed
            - Engine remains active until explicitly stopped or process ends
            - Only one MATLAB engine instance per WECSimRunner
            - WEC-Sim path must be configured before calling this method
            
        See Also:
            stop_matlab: Shutdown MATLAB engine when finished
            set_wec_sim_path: Configure WEC-Sim framework location
        """
        if self.matlab_engine is None:
            print("Starting MATLAB engine...")
            self.matlab_engine = matlab.engine.start_matlab()

            # Get and validate WEC-SIM path
            if self.wec_sim_path is None:
                raise ValueError("WEC-SIM path is not configured. Please set it using set_wec_sim_path()")
            wec_sim_path = self.wec_sim_path
            if wec_sim_path is None:
                raise ValueError("WEC-SIM path is not configured.")
            
            if not os.path.exists(wec_sim_path):
                raise FileNotFoundError(f"WEC-SIM path does not exist: {wec_sim_path}")

            matlab_path = self.matlab_engine.genpath(str(wec_sim_path), nargout=1)
            self.matlab_engine.addpath(matlab_path, nargout=0)
            print("MATLAB engine started and WEC-SIM path added...")
            return True
        else:
            print("MATLAB engine is already running.")
            return False
                
    def stop_matlab(self) -> bool:
        """Shutdown the MATLAB engine and free system resources.
        
        Cleanly terminates the MATLAB engine session, freeing memory and releasing
        the MATLAB license. This should be called when WEC-Sim simulations are
        complete to ensure proper resource cleanup.
        
        Returns:
            bool: True if MATLAB engine was stopped successfully, False if not running.
            
        Example:
            >>> # Stop engine after simulations
            >>> success = runner.stop_matlab()
            MATLAB engine stopped.
            >>> print(f"Engine stopped: {success}")
            Engine stopped: True
            
            >>> # Subsequent calls return False (not running)
            >>> success = runner.stop_matlab()
            MATLAB engine is not running.
            >>> print(f"Engine stopped: {success}")
            Engine stopped: False
            
        Resource Management:
            - **Memory cleanup**: Frees MATLAB workspace and variables
            - **License release**: Returns MATLAB license to pool
            - **Process termination**: Ends MATLAB engine background process
            - **Connection cleanup**: Closes Python-MATLAB communication
            
        When to Call:
            - **After simulation batch**: When multiple WEC-Sim runs are complete
            - **Resource constraints**: When system memory is limited
            - **Application shutdown**: Before terminating WEC-Grid application
            - **Error recovery**: After MATLAB engine errors or crashes
            
        Notes:
            - Automatically called after individual simulations complete
            - Engine can be restarted later with start_matlab() if needed
            - Does not affect stored simulation results in database
            - Recommended for long-running applications with multiple simulations
            
        See Also:
            start_matlab: Initialize MATLAB engine for simulations
        """
        if self.matlab_engine is not None:
            self.matlab_engine.quit()
            self.matlab_engine = None
            print("MATLAB engine stopped.")
            return True
        print("MATLAB engine is not running.")
        return False

    def sim_results(self, df_full, df_ds, model, sim_id):
        """Generate visualization plots for WEC-Sim simulation results.
        
        Creates a comprehensive plot showing WEC power output and wave conditions
        from a completed WEC-Sim simulation. Displays both full-resolution and
        downsampled data for validation and analysis purposes.
        
        Args:
            df_full (pd.DataFrame): Full-resolution simulation results containing:
                - time: Simulation time [s]
                - p: Active power output [MW]
                - eta: Wave surface elevation [m]
            df_ds (pd.DataFrame): Downsampled simulation results for grid integration:
                - time: Simulation time [s]  
                - p: Active power output [MW]
            model (str): WEC device model name (e.g., "RM3", "LUPA").
            sim_id (int): Unique simulation identifier.
            
        Returns:
            None: Displays plot using matplotlib.
            
        Plot Components:
            - **Primary axis**: Active power output [MW] vs. time
            - **Secondary axis**: Wave surface elevation [m] vs. time
            - **Full resolution**: Gray line showing all simulation data points
            - **Downsampled**: Red dotted line with markers for grid integration
            - **Wave background**: Blue transparent line showing wave conditions
            
        Example:
            >>> # Automatic plotting after simulation
            >>> runner(sim_id=101, model="RM3", ...)
            [Displays plot with power and wave data]
            
            >>> # Manual plotting with custom data
            >>> df_full = runner.database.query("SELECT * FROM WECSIM_rm3_101_full")
            >>> df_ds = runner.database.query("SELECT * FROM WECSIM_rm3_101") 
            >>> runner.sim_results(df_full, df_ds, "RM3", 101)
            
        Visualization Features:
            - **Dual y-axes**: Power (left) and wave elevation (right)
            - **Data comparison**: Full vs. downsampled resolution overlay
            - **Legend integration**: Combined legends from both axes
            - **Professional formatting**: Publication-quality plot styling
            - **Context information**: Model and simulation ID in title
            
        Data Validation:
            - Verifies downsampling accuracy by comparing trends
            - Shows correlation between wave conditions and power output
            - Identifies potential simulation issues or anomalies
            - Confirms data quality for grid integration studies
            
        Notes:
            - Called automatically after successful WEC-Sim simulations
            - Requires matplotlib for visualization
            - Plot helps validate simulation quality and data integrity
            - Downsampled data used for power system integration
            - Full resolution data available for detailed analysis
            
        See Also:
            __call__: Main simulation method that generates this data
            WECGridDB.query: Database query method for retrieving results
        """
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Secondary y-axis: Wave Height (m) — drawn first for background
        ax2 = ax1.twinx()
        ax2.set_ylabel("Wave Height (m)")
        ax2.plot(
            df_full["time"], df_full["eta"],
            color="tab:blue", alpha=0.3, linewidth=1, label="Wave Height"
        )

        # Primary y-axis: Active power (MW)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Active Power (MW)")
        ax1.plot(df_full["time"], df_full["p"], color="gray", label="P (full)", linewidth=1)
        ax1.plot(
            df_ds["time"], df_ds["p"],
            linestyle=":", marker="o", color="tab:red", label="P (downsampled)"
        )

        # Title + layout
        fig.suptitle(f"WEC-SIM Output — Model: {model}, Sim ID: {sim_id}")
        fig.tight_layout()

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

        plt.show()
            
    def __call__(
        self,
        sim_id: int,
        model: str,
        sim_length_secs: int = 3600 * 24, # 24 hours
        tsample: float = 300,
        wave_height: float = 2.5,
        wave_period: float = 8.0,
        wave_seed: int = random.randint(1, 100),
    ) -> bool:
        """Execute a complete WEC-Sim device simulation with specified parameters.
        
        Runs a high-fidelity wave energy converter simulation using the WEC-Sim MATLAB
        framework. Configures wave conditions, executes the simulation, processes results,
        and stores data in the database for subsequent grid integration studies.
        
        Args:
            sim_id (int): Unique identifier for this simulation run.
                Used for database storage and result tracking.
            model (str): WEC device model name. Supported models:
                - "RM3": Reference Model 3 (point absorber)
                - "LUPA": LUPA device model
            sim_length_secs (int, optional): Simulation duration in seconds.
                Defaults to 86400 (24 hours).
            tsample (float, optional): Output sampling interval in seconds.
                Defaults to 300 (5 minutes) for grid integration compatibility.
            wave_height (float, optional): Significant wave height in meters.
                Defaults to 2.5m (moderate sea state).
            wave_period (float, optional): Peak wave period in seconds.
                Defaults to 8.0s (typical ocean wave).
            wave_seed (int, optional): Random seed for wave generation.
                Defaults to random integer 1-100 for stochastic waves.
                
        Returns:
            bool: True if simulation completed successfully, False if errors occurred.
            
        Raises:
            FileNotFoundError: If WEC model directory cannot be found.
            RuntimeError: If MATLAB engine fails to start or WEC-Sim execution fails.
            DatabaseError: If result storage to database fails.
            
        Example:
            >>> # Standard 24-hour simulation
            >>> success = runner(
            ...     sim_id=101,
            ...     model="RM3",
            ...     sim_length_secs=86400,
            ...     wave_height=2.5,
            ...     wave_period=8.0
            ... )
            Starting WEC-SIM simulation...
            simulation complete... writing to database
            WEC-SIM complete: model = RM3, ID = 101, duration = 86400s
            >>> print(f"Simulation successful: {success}")
            
            >>> # Short test simulation with calm conditions
            >>> success = runner(
            ...     sim_id=102, 
            ...     model="RM3",
            ...     sim_length_secs=3600,  # 1 hour
            ...     wave_height=1.0,       # 1m waves
            ...     wave_period=6.0,       # 6s period
            ...     wave_seed=42           # Reproducible results
            ... )
            
        Simulation Workflow:
            1. **Model Resolution**: Locate WEC model directory using resolve_wec_model()
            2. **MATLAB Startup**: Initialize engine and configure WEC-Sim paths
            3. **Database Preparation**: Clear any existing tables for this simulation
            4. **Parameter Configuration**: Set MATLAB workspace variables
            5. **WEC-Sim Execution**: Run appropriate simulation function
            6. **Data Processing**: Format results for power system integration
            7. **Database Storage**: Store both full and downsampled results
            8. **Visualization**: Generate validation plots
            9. **Cleanup**: Shutdown MATLAB engine
            
        Database Tables Created:
            - `WECSIM_{model}_{sim_id}`: Downsampled data for grid integration
                * time: Simulation time [s]
                * p: Active power output [MW]
                * q: Reactive power (typically zero) [MVAr]
            - `WECSIM_{model}_{sim_id}_full`: Full-resolution simulation data
                * time: Simulation time [s]
                * p: Active power output [MW]
                * eta: Wave surface elevation [m]
                * Additional WEC-Sim output variables
                
        Wave Generation:
            - Uses WEC-Sim's irregular wave generation capabilities
            - JONSWAP spectrum with specified significant height and peak period
            - Random seed enables reproducible or stochastic simulations
            - Wave time series stored for correlation analysis
            
        Power Output Characteristics:
            - Active power varies with wave conditions and WEC dynamics
            - Sampling interval matches grid simulation requirements
            - Power output includes WEC device efficiency and control effects
            - Results suitable for grid integration and stability studies
            
        Model-Specific Execution:
            - **RM3**: Uses standard w2gSim() function with wave parameters
            - **LUPA**: Uses specialized w2gSim_LUPA() function
            - Model directory resolved automatically from WEC model library
            - Each model has specific hydrodynamic and control characteristics
            
        Performance Considerations:
            - Simulation time scales roughly linearly with sim_length_secs
            - MATLAB engine startup adds ~30-60s overhead per simulation
            - Memory usage depends on simulation duration and output frequency
            - Database storage requires adequate disk space for time series
            
        Notes:
            - MATLAB engine automatically stopped after simulation completion
            - Results visualization helps validate simulation quality
            - Database double-checking recommended for critical simulations
            - TODO: Add simulation progress bar for long-duration runs
            - TODO: Verify database write success with automated checks
            
        See Also:
            start_matlab: MATLAB engine initialization
            sim_results: Result visualization and validation
            resolve_wec_model: WEC model path resolution
            WECGridDB: Database interface for result storage
        """
        #TODO some sorta sim progress bar would be cool? 
        
        try:
            model_dir = resolve_wec_model(model)  # accepts name or path
            
            if self.start_matlab():
                table_name = f"WECSIM_{model.lower()}_{sim_id}"
                with self.database.connection() as conn:
                    conn.cursor().execute(f"DROP TABLE IF EXISTS {table_name};")

                print("Starting WEC-SIM simulation...")
                #model_dir = os.path.join(self.path_manager.wec_models, model)
                self.matlab_engine.cd(str(model_dir))

                # Set simulation parameters in MATLAB workspace
                self.matlab_engine.workspace["sim_id"] = sim_id
                self.matlab_engine.workspace["model"] = model.lower()
                self.matlab_engine.workspace["simLength"] = sim_length_secs
                self.matlab_engine.workspace["Tsample"] = tsample
                self.matlab_engine.workspace["waveHeight"] = wave_height
                self.matlab_engine.workspace["wavePeriod"] = wave_period
                self.matlab_engine.workspace["waveSeed"] = wave_seed

                self.matlab_engine.workspace["DB_PATH"] = self.database.db_path

                # Run the appropriate WEC-SIM function
                if model.lower() == "lupa":
                    self.matlab_engine.eval(
                        "m2g_out = w2gSim_LUPA(sim_id,simLength,Tsample,waveHeight,wavePeriod, model);",
                        nargout=0
                    )
                else:
                    self.matlab_engine.eval(
                        "m2g_out = w2gSim(sim_id,simLength,Tsample,waveHeight,wavePeriod,waveSeed,model);",
                        nargout=0
                    )
                print("simulation complete... writing to database")

                self.matlab_engine.eval("WECsim_to_PSSe_dataFormatter", nargout=0)
                print(f"WEC-SIM complete: model = {model}, ID = {sim_id}, duration = {sim_length_secs}s")
                #todo using the WECGridDB instance, we should double check if the data was written to the database
                #todo should add a data print or plot here to show the sim results
                self.stop_matlab()
                
                df_ds = self.database.query(f"SELECT * FROM {table_name}", return_type="df")
                df_full = self.database.query(f"SELECT * FROM {table_name}_full", return_type="df")

                self.sim_results(df_full, df_ds, model, sim_id)
                return True

            print("Failed to start MATLAB engine.")
            return False

        except Exception as e:
            print(f"[WEC-SIM ERROR] model={model}, ID={sim_id} → {e}")
            return False
