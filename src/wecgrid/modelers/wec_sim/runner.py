"""
WEC-Sim simulation runner for Wave Energy Converter device-level modeling.

This module provides the interface between WEC-Grid and WEC-Sim for high-fidelity
wave energy converter simulations using MATLAB engine integration.
"""

import os
import random
import json
import io

import matlab.engine
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from wecgrid.database.wecgrid_db import WECGridDB
#from wecgrid.util.wecgrid_pathmanager import WECGridPathManager
from wecgrid.util.resources import resolve_wec_model


# Inside wecsim_runner.py (at the top)
from dataclasses import dataclass

# Configuration file path
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_CURR_DIR, "wecsim_config.json")


class WECSimRunner:
    """Interface for running WEC-Sim device-level simulations via MATLAB engine.
    
    Provides Python interface to WEC-Sim MATLAB toolbox for high-fidelity wave energy 
    converter simulations. Manages MATLAB engine, executes WEC-Sim models, and stores
    results in WEC-Grid database.
        
    Attributes:
        wec_sim_path (str, optional): Path to WEC-Sim MATLAB installation.
        database (WECGridDB): Database interface for simulation data storage.
        matlab_engine (matlab.engine.MatlabEngine, optional): Active MATLAB engine.
        
    Example:
        >>> runner = WECSimRunner(database)
        >>> runner.set_wec_sim_path("/path/to/WEC-Sim")
        >>> wec_sim_id = runner(
        ...     model="RM3",
        ...     sim_length=3600,
        ...     dt=0.1,
        ...     spectrum_type="PM", 
        ...     wave_class="irregular",
        ...     wave_height=2.5,
        ...     wave_period=8.0
        ... )
        
    Notes:
        - Requires MATLAB license and WEC-Sim installation
        - MATLAB engine startup takes 30-60 seconds
        - Results stored in new wec_simulations and wec_power_results tables
        - Supports WEC models in src/wecgrid/models/wec_models
        
    TODO:
        - Add async MATLAB engine support for better performance
        - Implement batch simulation capabilities
    """
    def __init__(self, database: WECGridDB):
        """Initialize WEC-Sim runner with database connection.
        
        Args:
            database (WECGridDB): Database interface for simulation data storage.
                Must be connected and accessible for result storage.
                
        Notes:
            - MATLAB engine initialized on first simulation call
            - WEC-Sim path loaded from config file if available
        """
        self.wec_sim_path: Optional[str] = None
        self.database: WECGridDB = database
        self.matlab_engine: Optional[matlab.engine.MatlabEngine] = None
        
        # Try to load WEC-Sim path from config file
        self._load_config()
    
    def _load_config(self) -> None:
        """Load WEC-Sim configuration from JSON file."""
        try:
            if os.path.exists(_CONFIG_FILE):
                with open(_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.wec_sim_path = config.get('wec_sim_path')
                    if self.wec_sim_path:
                        #print(f"Loaded WEC-Sim path from config: {self.wec_sim_path}")
                        pass
        except Exception as e:
            print(f"Warning: Could not load WEC-Sim config: {e}")
    
    def _save_config(self) -> None:
        """Save WEC-Sim configuration to JSON file."""
        try:
            config = {'wec_sim_path': self.wec_sim_path}
            with open(_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Saved WEC-Sim configuration to: {_CONFIG_FILE}")
        except Exception as e:
            print(f"Warning: Could not save WEC-Sim config: {e}")
    
    def set_wec_sim_path(self, path: str) -> None:
        """Configure the WEC-Sim MATLAB framework installation path.
        
        Sets the path to the WEC-Sim MATLAB toolbox installation, which is required
        for running device-level wave energy converter simulations. The path is
        validated and automatically saved to configuration file for persistence.
        
        Args:
            path (str): Absolute path to WEC-Sim framework root directory.
                Should contain WEC-Sim MATLAB functions and initialization files.
                
        Returns:
            None: Sets internal wec_sim_path attribute and saves to config.
            
        Raises:
            FileNotFoundError: If specified path does not exist.
            
        Example:
            >>> runner.set_wec_sim_path(r"C:\path\to\WEC-Sim")
            Saved WEC-Sim configuration to: wecsim_config.json

        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"WEC-SIM path does not exist: {path}")
            
        self.wec_sim_path = path
        self._save_config()  # Automatically save to config file
        
    def get_wec_sim_path(self) -> Optional[str]:
        """Get the currently configured WEC-Sim path.
        
        Returns:
            str: Path to WEC-Sim installation if configured, None otherwise.
            
        Example:
            >>> path = runner.get_wec_sim_path()
            >>> print(f"WEC-Sim path: {path}")
        """
        return self.wec_sim_path
        
    def show_config(self) -> None:
        """Display current WEC-Sim configuration.
        
        Shows the current WEC-Sim path and configuration file location.
        Useful for troubleshooting configuration issues.
        """
        print(f"WEC-Sim Configuration:")
        print(f"  Path: {self.wec_sim_path or 'Not configured'}")
        print(f"  Config file: {_CONFIG_FILE}")
        print(f"  Config exists: {os.path.exists(_CONFIG_FILE)}")
        
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
            print(f"Starting MATLAB Engine... ", end='')
            self.matlab_engine = matlab.engine.start_matlab()
            print("MATLAB engine started.")

            # Get and validate WEC-SIM path
            if self.wec_sim_path is None:
                raise ValueError("WEC-SIM path is not configured. Please set it using set_wec_sim_path()")
            wec_sim_path = self.wec_sim_path
            if wec_sim_path is None:
                raise ValueError("WEC-SIM path is not configured.")
            
            if not os.path.exists(wec_sim_path):
                raise FileNotFoundError(f"WEC-SIM path does not exist: {wec_sim_path}")
            print(f"Adding WEC-SIM to path... ", end='')
            matlab_path = self.matlab_engine.genpath(str(wec_sim_path), nargout=1)
            self.matlab_engine.addpath(matlab_path, nargout=0)
            print("WEC-SIM path added.")
            
            self.out = io.StringIO()
            self.err = io.StringIO()
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
            self.out = None
            self.err = None
            return True
        print("MATLAB engine is not running.")
        return False

    def sim_results(self, df_power, model, wec_sim_id):
        """Generate visualization plots for WEC-Sim simulation results.
        
        Creates a comprehensive plot showing WEC power output and wave conditions
        from a completed WEC-Sim simulation using the new database schema.
        
        Args:
            df_power (pd.DataFrame): Power and wave data with columns:
                - time: Simulation time [s]
                - p: Active power output [w] 
                - eta: Wave surface elevation [m]
            model (str): WEC device model name (e.g., "RM3", "LUPA").
            wec_sim_id (int): Unique WEC simulation identifier from database.
            
        Returns:
            None: Displays plot using matplotlib.
            
        Plot Components:
            - **Primary axis**: Active power output [w] vs. time
            - **Secondary axis**: Wave surface elevation [m] vs. time
            - **Power output**: Red line showing WEC power generation
            - **Wave background**: Blue transparent line showing wave conditions
            
        Example:
            >>> # Automatic plotting after simulation
            >>> wec_sim_id = runner(model="RM3", ...)
            [Displays plot with power and wave data]
            
            >>> # Manual plotting with power data
            >>> power_data = engine.database.query(
            ...     "SELECT time_sec as time, p_w as p, wave_elevation_m as eta "
            ...     "FROM wec_power_results WHERE wec_sim_id = ?", 
            ...     params=(wec_sim_id,), return_type="df")
            >>> runner.sim_results(power_data, "RM3", wec_sim_id)
            
        Visualization Features:
            - **Dual y-axes**: Power (left) and wave elevation (right)
            - **Legend integration**: Combined legends from both axes
            - **Professional formatting**: Publication-quality plot styling
            - **Context information**: Model and simulation ID in title
            
        Data Validation:
            - Shows correlation between wave conditions and power output
            - Identifies potential simulation issues or anomalies
            - Confirms data quality for grid integration studies
            
        Notes:
            - Called automatically after successful WEC-Sim simulations
            - Requires matplotlib for visualization
            - Plot helps validate simulation quality and data integrity
            - Full resolution data from new database schema
            
        See Also:
            __call__: Main simulation method that generates this data
            WECGridDB.query: Database query method for retrieving results
        """
        
        if df_power.empty:
            print("No power data available for visualization")
            return
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Secondary y-axis: Wave elevation (m) — drawn first for background
        ax2 = ax1.twinx()
        ax2.set_ylabel("Wave Elevation (m)")
        if 'eta' in df_power.columns:
            ax2.plot(
                df_power["time"], df_power["eta"],
                color="tab:blue", alpha=0.3, linewidth=1, label="Wave Elevation"
            )

        # Primary y-axis: Active power W
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Active Power W")
        ax1.plot(df_power["time"], df_power["p"], color="tab:red", label="Power Output", linewidth=1.5)

        # Title + layout
        fig.suptitle(f"WEC-SIM Output — Model: {model}, WEC Sim ID: {wec_sim_id}")
        fig.tight_layout()

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

        plt.show()
            
    def __call__(
        self,
        model: str = 'RM3',
        model_path: str = None,  # New parameter for explicit model path
        sim_length: int = 3600 * 24, # 24 hours
        delta_time: float = 0.1,
        spectrum_type: str = 'PM',
        wave_class: str = 'irregular',
        wave_height: float = 2.5,
        wave_period: float = 8.0,
        wave_seed: int = random.randint(1, 100),
    ) -> Optional[int]:
        """Execute a complete WEC-Sim device simulation with specified parameters.
        
        Runs a high-fidelity wave energy converter simulation using the WEC-Sim MATLAB
        framework. Configures wave conditions, executes the simulation, processes results,
        and stores data in the database for subsequent grid integration studies.
        
        Args:
            model (str, optional): WEC device model name. Defaults to "RM3".
                Supported models:
                - "RM3": Reference Model 3 (point absorber)
                - "LUPA": LUPA device model
            model_path (str, optional): Explicit path to WEC model directory.
                If provided, overrides automatic model resolution. Should contain
                the Simulink model file and associated data files.
            sim_length (int, optional): Simulation duration in seconds.
                Defaults to 86400 (24 hours).
            dt (float, optional): Simulation time step in seconds.
                Defaults to 0.1 for high-resolution WEC dynamics.
            spectrum_type (str, optional): Wave spectrum type.
                Defaults to 'PM' (Pierson-Moskowitz). Options: 'PM', 'JONSWAP', etc.
            wave_class (str, optional): Wave type classification.
                Defaults to 'irregular'. Options: 'irregular', 'regular', etc.
            wave_height (float, optional): Significant wave height in meters.
                Defaults to 2.5m (moderate sea state).
            wave_period (float, optional): Peak wave period in seconds.
                Defaults to 8.0s (typical ocean wave).
            wave_seed (int, optional): Random seed for wave generation.
                Defaults to random integer 1-100 for stochastic waves.
                
        Returns:
            int: wec_sim_id from database if successful, None if failed.
            
        Raises:
            FileNotFoundError: If WEC model directory cannot be found.
            RuntimeError: If MATLAB engine fails to start or WEC-Sim execution fails.
            DatabaseError: If result storage to database fails.
            
        Example:
            >>> # Standard 24-hour simulation with Pierson-Moskowitz spectrum
            >>> wec_sim_id = runner(
            ...     model="RM3",
            ...     sim_length=86400,
            ...     dt=0.1,
            ...     spectrum_type="PM",
            ...     wave_class="irregular",
            ...     wave_height=2.5,
            ...     wave_period=8.0
            ... )
            Starting WEC-SIM simulation...
            simulation complete... writing to database
            WEC-SIM complete: model = RM3, wec_sim_id = 42, duration = 86400s
            >>> print(f"WEC simulation ID: {wec_sim_id}")
            
            >>> # Short test simulation with JONSWAP spectrum
            >>> wec_sim_id = runner(
            ...     model="RM3",
            ...     sim_length=3600,       # 1 hour
            ...     dt=0.05,               # Fine time step
            ...     spectrum_type="JONSWAP",
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
            - `wec_simulations`: WEC simulation metadata
                * wec_sim_id: Unique simulation identifier
                * model_type: WEC model name (e.g., "RM3", "LUPA")
                * sim_duration_sec: Simulation duration [s]
                * delta_time: Simulation time step [s]
                * wave_height_m: Significant wave height [m]
                * wave_period_sec: Peak wave period [s]
                * wave_spectrum: Wave spectrum type (e.g., "PM", "JONSWAP")
                * wave_class: Wave classification (e.g., "irregular", "regular")
                * wave_seed: Random seed for wave generation
                * simulation_hash: Unique hash for simulation parameters
            - `wec_power_results`: High-resolution power time-series
                * wec_sim_id: Foreign key to wec_simulations
                * time_sec: Simulation time [s]
                * device_index: WEC device number (1 for single device)
                * p_w: Active power output [W]
                * q_var: Reactive power output [VAr]
                * wave_elevation_m: Wave surface elevation [m]
                
        Wave Generation:
            - Uses WEC-Sim's configurable wave generation capabilities
            - Multiple spectrum types: PM, JONSWAP, etc.
            - Supports both regular and irregular wave classes
            - Random seed enables reproducible or stochastic simulations
            - Wave time series stored for correlation analysis
            
        Notes:
            - Results stored in new wec_simulations and wec_power_results tables
            - MATLAB engine automatically stopped after simulation completion
            - Results visualization helps validate simulation quality
            - Power values stored in Watts (W) as per new database schema
            - No downsampling performed - downsampling done later in WEC-Grid
        
        TODO:
            - Add simulation progress bar for long-duration runs
            - Verify database write success with automated checks
            - Add support for multi-device WEC farms 
            
        """
        print(r"""
              
            ⠀ WEC-SIM⠀⠀⠀⠀     ⣠⣴⣶⠾⠿⠿⠯⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣾⠛⠁⠀⠀⠀⠀⠀⠀⠈⢻⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⠿⠁⠀⠀⠀⢀⣤⣾⣟⣛⣛⣶⣬⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠟⠃⠀⠀⠀⠀⠀⣾⣿⠟⠉⠉⠉⠉⠛⠿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⡟⠋⠀⠀⠀⠀⠀⠀⠀⣿⡏⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⣠⡿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣷⡍⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣤⣤⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⣠⣼⡏⠀⠀           ⠈⠙⠷⣤⣤⣠⣤⣤⡤⡶⣶⢿⠟⠹⠿⠄⣿⣿⠏⠀⣀⣤⡦⠀⠀⠀⠀⣀⡄
            ⢀⣄⣠⣶⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠓⠚⠋⠉⠀⠀⠀⠀⠀⠀⠈⠛⡛⡻⠿⠿⠙⠓⢒⣺⡿⠋⠁
            ⠉⠉⠉⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀
            """)
        
        try:
            # Use provided model_path if available, otherwise use old resolution logic
            if model_path:
                # Use the explicitly provided model path
                model_dir = model_path
                model_name = model  # Keep the model name as provided
                if not os.path.exists(model_dir):
                    raise FileNotFoundError(f"WEC model path '{model_path}' does not exist")
            elif os.path.isabs(model) and os.path.exists(model):
                # If it's already an absolute path and exists, use it
                model_dir = model
                model_name = os.path.basename(model)
            else:
                # Look for the model in the data directory (outside src)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir))))
                model_dir = os.path.join(repo_root, 'data', 'wec_models', model)
                model_name = model
                
                if not os.path.exists(model_dir):
                    raise FileNotFoundError(f"WEC model '{model}' not found at {model_dir}")
            
            if self.start_matlab():
                print("Starting WEC-SIM simulation...")
                print(f"\t Model: {model_name}\n"
                      f"\t Model Path: {model_dir}\n"
                      f"\t Simulation Length: {sim_length} seconds\n"
                      f"\t Time Step: {delta_time} seconds\n"
                      f"\t Wave class: {wave_class}\n"
                      f"\t Wave Height: {wave_height} m\n"
                      f"\t Wave Period: {wave_period} s\n"
                      )

                current_dir = os.path.dirname(os.path.abspath(__file__))
                self.matlab_engine.addpath(current_dir, nargout=0)  # add the directory containing this file
                self.matlab_engine.addpath(model_dir, nargout=0)

                # Set simulation parameters in MATLAB workspace
                self.matlab_engine.workspace["simLength"] = sim_length
                self.matlab_engine.workspace["dt"] = delta_time
                self.matlab_engine.workspace["spectrumType"] = spectrum_type
                self.matlab_engine.workspace["waveClassType"] = wave_class
                self.matlab_engine.workspace["waveHeight"] = wave_height
                self.matlab_engine.workspace["wavePeriod"] = wave_period
                self.matlab_engine.workspace["waveSeed"] = int(wave_seed)
                self.matlab_engine.workspace["wecModel"] = model_name  # Pass the WEC model name
                #print(f"Debug: {model_dir}")
                self.matlab_engine.workspace["wecModelPath"] = str(model_dir)  # Pass resolved model path
                # self.matlab_engine.workspace["source_files"] = current_dir

                self.matlab_engine.workspace["DB_PATH"] = self.database.db_path

                # Run the unified WEC-SIM function - MATLAB can find it since we added the path
                self.matlab_engine.eval(
                    "[m2g_out] = w2gSim(simLength,dt,spectrumType,waveClassType,waveHeight,wavePeriod,waveSeed);",
                    nargout=0
                )
                print("simulation complete... writing to database")

                # Run the formatter script - MATLAB can find it since we added the path
                self.matlab_engine.eval("run('formatter.m')", nargout=0)
                
                # Get the wec_sim_id that was created by the database
                wec_sim_id = self.matlab_engine.workspace["wec_sim_id_result"]
                wec_sim_id = int(wec_sim_id)  # Convert from MATLAB double to Python int
                
                print(f"WEC-SIM complete: model = {model_name}, wec_sim_id = {wec_sim_id}, duration = {sim_length}s")
                
                # Query power results for visualization using the returned wec_sim_id
                power_query = """
                    SELECT time_sec as time, p_w as p, wave_elevation_m as eta 
                    FROM wec_power_results 
                    WHERE wec_sim_id = ? 
                    ORDER BY time_sec
                """
                df_power = self.database.query(
                    power_query, 
                    params=(wec_sim_id,), 
                    return_type="df"
                )
                
                self.stop_matlab()
                
                if not df_power.empty:
                    self.sim_results(df_power, model, wec_sim_id)
                
                return wec_sim_id

            print("Failed to start MATLAB engine.")
            return None

        except Exception as e:
            print(f"[WEC-SIM ERROR] model={model} → {e}")
            self.stop_matlab()
            return None
