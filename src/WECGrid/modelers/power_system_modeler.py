# power_system_modeler.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime        # <- use datetime to match PSSEModeler
import pandas as pd
from .grid_state import GridState
from ..wec.wecfarm import WECFarm


class PowerSystemModeler(ABC):
    """Abstract base class for power system modeling interfaces.
    
    This class defines the common interface for power system modeling backends
    in the WEC-GRID framework. It provides a standardized API for grid analysis,
    WEC farm integration, and time-series simulation that can be implemented
    for different power system analysis tools (PSS®E, PyPSA, etc.).
    
    The PowerSystemModeler establishes a common pattern for loading case files,
    solving power flows, modifying grid components, and capturing simulation
    results. All concrete implementations must provide the abstract methods
    while inheriting the common grid state management functionality.
    
    Args:
        engine (Any): The WEC-GRID simulation engine containing case configuration,
            time management, and WEC farm definitions. Must have attributes for
            case_file, time, and wec_farms.
    
    Attributes:
        engine (Any): Reference to the simulation engine.
        grid (GridState): Current grid state containing time-series data for all
            components (buses, generators, lines, loads).
        sbase (Optional[float]): System base MVA from the power system case [MVA].
        
    Example:
        >>> # This is an abstract class - use concrete implementations
        >>> from wecgrid.modelers import PSSEModeler, PyPSAModeler
        >>> psse_model = PSSEModeler(engine)
        >>> pypsa_model = PyPSAModeler(engine)
        
    Notes:
        - This is an abstract base class and cannot be instantiated directly
        - All abstract methods must be implemented by concrete subclasses
        - Provides common grid state management and property accessors
        - Supports cross-platform validation between different modeling tools
        - Grid state data follows standardized schema for component comparison
        
    See Also:
        PSSEModeler: PSS®E-specific implementation
        PyPSAModeler: PyPSA-specific implementation
        GridState: Grid state management and time-series storage
    """
    
    def __init__(self, engine: Any):
        """Initialize the PowerSystemModeler with the simulation engine.
        
        Creates a new PowerSystemModeler instance and sets up the basic modeling
        framework. The engine object must contain case file information, time
        configuration, and WEC farm definitions.
        
        Args:
            engine (Any): WEC-GRID simulation engine with the following required attributes:
                - case_file (str): Path to power system case file
                - time: Time management object with start_time and snapshots
                - wec_farms (List[WECFarm]): List of WEC farm objects for integration
                
        Note:
            This method only performs basic initialization. Concrete implementations
            must call their specific ``init_api()`` method to initialize the backend
            power system analysis tool.
        """
        self.engine = engine
        self.grid = GridState()
        self.sbase: Optional[float] = None
        

    @abstractmethod
    def init_api(self) -> bool:
        """Initialize the backend power system analysis API and load the case.
        
        This method must set up the specific power system analysis tool (PSS®E, PyPSA, etc.),
        load the case file, and perform initial power flow solution. It should also take
        an initial snapshot of the grid state.
        
        Returns:
            bool: True if initialization is successful, False otherwise.
            
        Raises:
            ImportError: If the backend power system tool is not found or not configured.
            ValueError: If the case file cannot be loaded or is invalid.
            
        Notes:
            Implementations should:
            
            - Initialize the backend API/environment
            - Load the specified case file (.sav, .raw, etc.)
            - Set the system base MVA (self.sbase) [MVA]
            - Perform initial power flow solution
            - Take initial grid state snapshot
            - Handle any backend-specific configuration
            
        Example:
            >>> modeler = ConcreteModeler(engine)
            >>> if modeler.init_api():
            ...     print("Backend initialized successfully")
            ... else:
            ...     print("Failed to initialize backend")
        """
        pass

    @abstractmethod
    def solve_powerflow(self) -> bool:
        """Run a power flow solution using the backend solver.
        
        Executes the power flow calculation using the specific backend tool's solver
        and verifies that the solution converged successfully.
        
        Returns:
            bool: True if power flow converged, False otherwise.
            
        Notes:
            Implementations should:
            
            - Call the backend's power flow solver
            - Check convergence status
            - Handle any solver-specific parameters
            - Report convergence failures appropriately
            - Optionally suppress verbose solver output
            
        Example:
            >>> if modeler.solve_powerflow():
            ...     print("Power flow converged")
            ... else:
            ...     print("Power flow failed to converge")
        """
        pass

    @abstractmethod
    def add_wec_farm(self, farm: WECFarm) -> bool:
        """Add a WEC farm to the power system model.

        Creates the necessary electrical infrastructure to integrate a WEC farm
        into the existing power system model, including buses, generators, and
        transmission connections.

        Args:
            farm (WECFarm): The WEC farm object containing connection details,
                power characteristics, and identification information.

        Returns:
            bool: True if the farm is added successfully, False otherwise.

        Raises:
            ValueError: If the WEC farm cannot be added due to invalid parameters.
            
        Notes:
            Implementations should:
            
            - Create new bus for WEC farm connection
            - Add WEC generator with appropriate power characteristics
            - Create transmission line to existing grid
            - Handle backend-specific component creation
            - Update grid state after modifications
            - Solve power flow to validate changes
            
        Example:
            >>> wec_farm = WECFarm(bus_location=999, connecting_bus=14)
            >>> if modeler.add_wec_farm(wec_farm):
            ...     print("WEC farm added successfully")
        """
        pass

    @abstractmethod
    def simulate(self,
                 load_curve: Optional[pd.DataFrame] = None) -> bool:
        """Run time-series simulation with WEC farm and load updates.
        
        Performs dynamic simulation over multiple time snapshots, updating WEC farm
        generator outputs and optionally bus loads at each time step.
        
        Args:
            load_curve (Optional[pd.DataFrame]): DataFrame containing load values for 
                each bus at each snapshot. Index should be snapshots, columns should 
                be bus IDs. If None, loads remain constant throughout simulation.

        Returns:
            bool: True if the simulation completes successfully, False otherwise.
            
        Raises:
            Exception: If there is an error updating components or solving power flow
                at any snapshot.

        Notes:
            Implementations should:
            
            - Iterate through all time snapshots from engine.time
            - Update WEC generator power outputs [MW] from farm power curves
            - Update bus loads [MW] if load_curve is provided
            - Solve power flow at each time step
            - Capture grid state snapshots for analysis
            - Provide progress indication for long simulations
            - Handle any convergence failures gracefully
            
        Example:
            >>> # Simulate with constant loads
            >>> modeler.simulate()
            >>> 
            >>> # Simulate with time-varying loads
            >>> load_df = pd.DataFrame(load_data, index=snapshots, columns=bus_ids)
            >>> modeler.simulate(load_curve=load_df)
        """
        pass

    @abstractmethod
    def take_snapshot(self, timestamp: datetime) -> None:
        """Capture and store current grid state at the specified timestamp.
        
        Extracts the current state of all grid components (buses, generators, lines,
        and loads) from the backend power system tool and updates the internal grid
        state object with time-series data.
        
        Args: 
            timestamp (datetime): The timestamp for the snapshot.

        Returns:
            None
            
        Notes:
            Implementations should:
            
            - Extract bus data: voltages [pu], [degrees], power injections [MW], [MVAr]
            - Extract generator data: power outputs [MW], [MVAr], status
            - Extract line data: power flows [MW], [MVAr], loading [%]
            - Extract load data: power consumption [MW], [MVAr]
            - Convert all data to standardized WEC-GRID schema format
            - Store data in self.grid with proper timestamp indexing
            - Ensure consistent component naming across backends
            
        Example:
            >>> from datetime import datetime
            >>> timestamp = datetime.now()
            >>> modeler.take_snapshot(timestamp)
            >>> # Grid state now contains data at the specified timestamp
        """
        pass

    # Convenience accessors
    @property
    def bus(self) -> Optional[pd.DataFrame]:
        """Get the latest bus state DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing current bus state with columns:
                bus, bus_name, type, p, q, v_mag, angle_deg, base.
                Returns None if no snapshots have been taken.
                
        Note:
            All power values are in per-unit on system base MVA.
            Voltage magnitudes are in per-unit, angles in degrees.
        """
        return self.grid.bus

    @property
    def gen(self) -> Optional[pd.DataFrame]:
        """Get the latest generator state DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing current generator state with columns:
                gen, bus, p, q, base, status.
                Returns None if no snapshots have been taken.
                
        Note:
            All power values are in per-unit on generator base MVA.
            Status indicates generator availability (1=online, 0=offline).
        """
        return self.grid.gen

    @property
    def load(self) -> Optional[pd.DataFrame]:
        """Get the latest load state DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing current load state with columns:
                load, bus, p, q, base, status.
                Returns None if no snapshots have been taken.
                
        Note:
            All power values are in per-unit on system base MVA.
            Status indicates load connectivity (1=connected, 0=disconnected).
        """
        return self.grid.load

    @property
    def line(self) -> Optional[pd.DataFrame]:
        """Get the latest transmission line state DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing current line state with columns:
                line, ibus, jbus, line_pct, status.
                Returns None if no snapshots have been taken.
                
        Note:
            Line loading (line_pct) is expressed as percentage of thermal rating.
            Status indicates line availability (1=in-service, 0=out-of-service).
        """
        return self.grid.line

    @property
    def bus_t(self) -> Dict[str, pd.DataFrame]:
        """Get time-series bus data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with timestamp strings as keys and
                bus state DataFrames as values. Each DataFrame contains bus data
                for the corresponding snapshot timestamp.
                
        Note:
            Provides historical bus state data for time-series analysis and plotting.
            All power values are in per-unit on system base MVA.
        """
        return self.grid.bus_t

    @property
    def gen_t(self) -> Dict[str, pd.DataFrame]:
        """Get time-series generator data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with timestamp strings as keys and
                generator state DataFrames as values. Each DataFrame contains generator
                data for the corresponding snapshot timestamp.
                
        Note:
            Provides historical generator state data for time-series analysis and plotting.
            All power values are in per-unit on generator base MVA.
        """
        return self.grid.gen_t

    @property
    def load_t(self) -> Dict[str, pd.DataFrame]:
        """Get time-series load data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with timestamp strings as keys and
                load state DataFrames as values. Each DataFrame contains load data
                for the corresponding snapshot timestamp.
                
        Note:
            Provides historical load state data for time-series analysis and plotting.
            All power values are in per-unit on system base MVA.
        """
        return self.grid.load_t

    @property
    def line_t(self) -> Dict[str, pd.DataFrame]:
        """Get time-series transmission line data for all snapshots.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with timestamp strings as keys and
                line state DataFrames as values. Each DataFrame contains line data
                for the corresponding snapshot timestamp.
                
        Note:
            Provides historical line state data for time-series analysis and plotting.
            Line loading percentages indicate utilization of thermal capacity.
        """
        return self.grid.line_t