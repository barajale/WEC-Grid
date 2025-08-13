# power_system_modeler.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime        # <- use datetime to match PSSEModeler
import pandas as pd
from .grid_state import GridState
from ..wec.wecfarm import WECFarm


class PowerSystemModeler(ABC):
    """Abstract base class for power system modeling backends.
    
    Defines standardized interface for PSS®E, PyPSA, and other power system tools
    in WEC-GRID framework. Provides grid analysis, WEC integration, and time-series
    simulation capabilities through common API.
    
    Args:
        engine: WEC-GRID Engine with case_file, time, and wec_farms attributes.
    
    Attributes:
        engine: Reference to simulation engine.
        grid (GridState): Time-series data for buses, generators, lines, loads.
        sbase (float, optional): System base power [MVA].
        
    Example:
        >>> from wecgrid.modelers import PSSEModeler, PyPSAModeler
        >>> psse_model = PSSEModeler(engine)
        >>> pypsa_model = PyPSAModeler(engine)
        
    Notes:
        - Abstract class - use concrete implementations (PSSEModeler, PyPSAModeler)
        - Grid state data follows standardized schema for cross-platform comparison
        - All abstract methods must be implemented by subclasses
    """
    
    def __init__(self, engine: Any):
        """Initialize PowerSystemModeler with simulation engine.
        
        Args:
            engine: WEC-GRID Engine with case_file, time, and wec_farms attributes.
                
        Note:
            Call init_api() after construction to initialize backend tool.
        """
        self.engine = engine
        self.grid = GridState()
        self.sbase: Optional[float] = None
        

    @abstractmethod
    def init_api(self) -> bool:
        """Initialize backend power system tool and load case file.
        
        Returns:
            bool: True if initialization successful, False otherwise.
            
        Raises:
            ImportError: If backend tool not found or configured.
            ValueError: If case file invalid or cannot be loaded.
            
        Notes:
            Implementation should:
            
            - Initialize backend API/environment
            - Load case file (.sav, .raw, etc.)
            - Set system base MVA (self.sbase)
            - Perform initial power flow solution
            - Take initial grid state snapshot
            
        Example:
            >>> if modeler.init_api():
            ...     print("Backend initialized successfully")
        """
        pass

    @abstractmethod
    def solve_powerflow(self) -> bool:
        """Run power flow solution using backend solver.
        
        Returns:
            bool: True if power flow converged, False otherwise.
            
        Notes:
            Implementation should:
            
            - Call backend's power flow solver
            - Check convergence status
            - Handle solver-specific parameters
            - Suppress verbose output if needed
            
        Example:
            >>> if modeler.solve_powerflow():
            ...     print("Power flow converged")
        """
        pass

    @abstractmethod
    def add_wec_farm(self, farm: WECFarm) -> bool:
        """Add WEC farm to power system model.

        Args:
            farm (WECFarm): WEC farm with connection details and power characteristics.

        Returns:
            bool: True if farm added successfully, False otherwise.

        Raises:
            ValueError: If WEC farm parameters invalid.
            
        Notes:
            Implementation should:
            
            - Create new bus for WEC connection
            - Add WEC generator with power characteristics
            - Create transmission line to existing grid
            - Update grid state after modifications
            - Solve power flow to validate changes
            
        Example:
            >>> if modeler.add_wec_farm(wec_farm):
            ...     print("WEC farm added successfully")
        """
        pass

    @abstractmethod
    def simulate(self, load_curve: Optional[pd.DataFrame] = None) -> bool:
        """Run time-series simulation with WEC and load updates.
        
        Args:
            load_curve (pd.DataFrame, optional): Load values for each bus at each snapshot.
                Index: snapshots, columns: bus IDs. If None, loads remain constant.

        Returns:
            bool: True if simulation completes successfully, False otherwise.
            
        Raises:
            Exception: If error updating components or solving power flow.

        Notes:
            Implementation should:
            
            - Iterate through all time snapshots from engine.time
            - Update WEC generator power outputs [MW] from farm data
            - Update bus loads [MW] if load_curve provided
            - Solve power flow at each time step
            - Capture grid state snapshots for analysis
            - Handle convergence failures gracefully
            
        Example:
            >>> # Constant loads
            >>> modeler.simulate()
            >>> 
            >>> # Time-varying loads
            >>> modeler.simulate(load_curve=load_df)
        """
        pass

    @abstractmethod
    def take_snapshot(self, timestamp: datetime) -> None:
        """Capture current grid state at specified timestamp.
        
        Args: 
            timestamp (datetime): Timestamp for the snapshot.

        Notes:
            Implementation should:
            
            - Extract bus data: voltages [p.u.], [degrees], power [MW], [MVAr]
            - Extract generator data: power outputs [MW], [MVAr], status
            - Extract line data: power flows [MW], [MVAr], loading [%]
            - Extract load data: power consumption [MW], [MVAr]
            - Convert to standardized WEC-GRID schema
            - Store in self.grid with timestamp indexing
            
        Example:
            >>> modeler.take_snapshot(datetime.now())
        """
        pass

    # Convenience accessors
    @property
    def bus(self) -> Optional[pd.DataFrame]:
        """Current bus state with columns: bus, bus_name, type, p, q, v_mag, angle_deg, base.
        
        Returns:
            pd.DataFrame: Bus state data [p.u. on system MVA base] or None if no snapshots.
        """
        return self.grid.bus

    @property
    def gen(self) -> Optional[pd.DataFrame]:
        """Current generator state with columns: gen, bus, p, q, base, status.
        
        Returns:
            pd.DataFrame: Generator state data [p.u. on generator MVA base] or None if no snapshots.
        """
        return self.grid.gen

    @property
    def load(self) -> Optional[pd.DataFrame]:
        """Current load state with columns: load, bus, p, q, base, status.
        
        Returns:
            pd.DataFrame: Load state data [p.u. on system MVA base] or None if no snapshots.
        """
        return self.grid.load

    @property
    def line(self) -> Optional[pd.DataFrame]:
        """Current line state with columns: line, ibus, jbus, line_pct, status.
        
        Returns:
            pd.DataFrame: Line state data [line_pct as % of thermal rating] or None if no snapshots.
        """
        return self.grid.line

    @property
    def bus_t(self) -> Dict[str, pd.DataFrame]:
        """Time-series bus data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Keys: timestamp strings, Values: bus state DataFrames.
        """
        return self.grid.bus_t

    @property
    def gen_t(self) -> Dict[str, pd.DataFrame]:
        """Time-series generator data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Keys: timestamp strings, Values: generator state DataFrames.
        """
        return self.grid.gen_t

    @property
    def load_t(self) -> Dict[str, pd.DataFrame]:
        """Time-series load data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Keys: timestamp strings, Values: load state DataFrames.
        """
        return self.grid.load_t

    @property
    def line_t(self) -> Dict[str, pd.DataFrame]:
        """Time-series line data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Keys: timestamp strings, Values: line state DataFrames.
        """
        return self.grid.line_t