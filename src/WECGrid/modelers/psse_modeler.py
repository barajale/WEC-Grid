"""
PSS®E Modeler
"""

# Standard Libraries
import os
import sys
import contextlib
from typing import Any, List, Optional, Dict
from datetime import datetime
from collections import defaultdict

# 3rd party
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Local imports
from .power_system_modeler import PowerSystemModeler
from .grid_state import GridState  # used internally

from ..wec.wecfarm import WECFarm


@contextlib.contextmanager
def silence_stdout():
    """Context manager to suppress stdout output.
    
    This utility function redirects stdout to devnull temporarily, which is useful
    for suppressing verbose output from the PSS®E Python API during initialization.
    
    Yields:
        None: Context where stdout is suppressed.
        
    Example:
        with silence_stdout():
            # PSS®E API calls with suppressed output
            psspy.psseinit(50)
    """
    new_target = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = new_target
    try:
        yield
    finally:
        sys.stdout = old_stdout
        
class PSSEModeler(PowerSystemModeler):
    """PSS®E power system modeling interface.
    
    This class provides a comprehensive interface for power system modeling and simulation
    using Siemens PSS®E software. It inherits from the PowerSystemModeler abstract base
    class and implements PSS®E-specific functionality for grid analysis, WEC farm integration,
    and time-series simulation.
    
    The PSSEModeler handles PSS®E case file loading, power flow solutions, component
    modifications, and data extraction. It supports both .sav (saved case) and .raw
    (raw data) file formats and provides methods for dynamic simulation with WEC farms.
    
    Args:
        engine (Any): The WEC-GRID simulation engine containing case configuration,
            time management, and WEC farm definitions. Must have attributes for
            case_file, time, and wec_farms.
    
    Attributes:
        engine (Any): Reference to the simulation engine.
        grid (GridState): Current grid state containing time-series data for all
            components (buses, generators, lines, loads).
        sbase (float): System base MVA from PSS®E case [MVA].
        psspy (module): PSS®E Python API module for direct API access.
        
    Example:

        
    Notes:
        - Requires PSS®E software installation and valid license
        - Compatible with PSS®E version 35.3 Python API
        - Automatically removes reactive power limits from generators for PyPSA compatibility
        - Supports time-series simulation with configurable load curves
        - Grid state is captured at each simulation snapshot for analysis
        
    Raises:
        ImportError: If PSS®E software is not found or not properly configured.
        ValueError: If the provided engine lacks required attributes.
    """
    def __init__(self, engine: Any):
        """Initialize the PSSEModeler with the simulation engine.
        
        Creates a new PSSEModeler instance and calls the parent PowerSystemModeler
        constructor to set up the basic modeling framework. The engine object must
        contain case file information, time configuration, and WEC farm definitions.
        
        Args:
            engine (Any): WEC-GRID simulation engine with the following required attributes:
                - case_file (str): Path to PSS®E case file (.sav or .raw)
                - time: Time management object with start_time and snapshots
                - wec_farms (List[WECFarm]): List of WEC farm objects for integration
                
        Note:
            This method only performs basic initialization. Call ``init_api()`` after
            instantiation to initialize the PSS®E environment and load the case file.
            
        Example:

        """
        super().__init__(engine)

    def __repr__(self) -> str:
        """Return a formatted string representation of the PSSEModeler.
        
        Provides a tree-style summary of the PSS®E model including the case name,
        component counts, and system base MVA. This representation gives users
        a quick overview of the loaded grid model structure.
        
        Returns:
            str: Multi-line string representation showing:
                - Case file name
                - Number of buses, generators, loads, and lines
                - System base MVA [MVA]
                
        Example:
            >>> print(modeler)
            psse:
            ├─ case: IEEE_14_bus.sav
            ├─ buses: 14
            ├─ generators: 5
            ├─ loads: 11
            └─ lines: 20
            
            Sbase: 100.0 MVA
            
        Note:
            This method requires that ``init_api()`` has been called successfully
            and that grid state data is available.
        """
        return (
            f"psse:\n"
            f"├─ case: {self.engine.case_name}\n"
            f"├─ buses: {len(self.grid.bus)}\n"
            f"├─ generators: {len(self.grid.gen)}\n"
            f"├─ loads: {len(self.grid.load)}\n"
            f"└─ lines: {len(self.grid.line)}"
            f"\n"
            f"Sbase: {self.sbase} MVA"
        )
        
    def init_api(self) -> bool:
        """Initialize the PSS®E environment and load the case.
        
        This method sets up the PSS®E Python API, loads the specified case file,
        and performs initial power flow solution. It also removes reactive power
        limits on generators and takes an initial snapshot.
        
        Returns:
            bool: True if initialization is successful, False otherwise.
            
        Raises:
            ImportError: If PSS®E is not found or not configured correctly.
            
        Notes:
            The following PSS®E API calls are used for initialization:
            
            - ``psseinit()``: Initialize PSS®E environment
            - ``case()`` or ``read()``: Load case file (.sav or .raw)
            - ``sysmva()``: Get system MVA base
                Returns: System base MVA [MVA]
            - ``fnsl()``: Solve power flow
            - ``solved()``: Check solution status
                Returns: 0 = converged, 1 = not converged [dimensionless]
        """
        Debug = False  # Set to True for debugging output
        try:
            with silence_stdout():
                import pssepath # TODO double check this works, conda work around might not be needed
                pssepath.add_pssepath()
                import psspy
                import psse35
                import redirect
                redirect.psse2py() 
                psse35.set_minor(3)
                psspy.psseinit(50)

            if not Debug:
                psspy.prompt_output(6, "", [])
                psspy.alert_output(6, "", [])
                psspy.progress_output(6, "", [])

            PSSEModeler.psspy = psspy
            self._i = psspy.getdefaultint()
            self._f = psspy.getdefaultreal()
            self._s = psspy.getdefaultchar()
            

        except ModuleNotFoundError as e:
            raise ImportError("PSS®E not found or not configured correctly.") from e

        ext = self.engine.case_file.lower()
        if ext.endswith(".sav"):
            ierr = psspy.case(self.engine.case_file)
        elif ext.endswith(".raw"):
            ierr = psspy.read(1, self.engine.case_file)
        else:
            print("Unsupported case file format.")
            return False

        if ierr != 0:
            print(f"PSS®E failed to load case. ierr={ierr}")
            return False
    
        self.sbase = self.psspy.sysmva()
        if not self.solve_powerflow(): # true is good, false is failed power flow
            print("Powerflow solution failed.")
            return False
        
        self.adjust_reactive_lim()  # Remove reactive limits on generators
        self.take_snapshot(timestamp=self.engine.time.start_time)
        print("PSS®E software initialized")
        return True

    def solve_powerflow(self) -> bool:
        """Run power flow solution and check convergence.
        
        Executes the PSS®E power flow solver using the Newton-Raphson method
        and verifies that the solution converged successfully.
        
        Returns:
            bool: True if power flow converged, False otherwise.
            
        Notes:
            The following PSS®E API calls are used:
            
            - ``fnsl()``: Full Newton-Raphson power flow solution
            - ``solved()``: Check if power flow solution converged (0 = converged)
        """
        ierr = self.psspy.fnsl()
        ival = self.psspy.solved()
        if ierr != 0 or ival != 0:
            print(f"[ERROR] Powerflow not solved. PSS®E error code: {ierr}, Solved Status: {ival}")
            #TODO error handling here 
            return False
        return True
        #TODO not sure if I should be calling take_snapshot here or in simulate? maybe both?

    def adjust_reactive_lim(self) -> bool:
        """Remove reactive power limits from all generators.
        
        Adjusts all generators in the PSS®E case to remove reactive power limits
        by setting QT = +9999 and QB = -9999. This is used to more closely align 
        the modeling behavior between PSS®E and PyPSA.
        
        Returns:
            bool: True if successful, False otherwise.
            
        Notes:
            The following PSS®E API calls are used:
            
            - ``amachint()``: Get all generator bus numbers
                Returns: Bus numbers [dimensionless]
            - ``machine_chng_4()``: Modify generator reactive power limits
                - Sets QT (Q max) to 9999.0 [MVAr]
                - Sets QB (Q min) to -9999.0 [MVAr]
        """
        ierr, gen_buses = self.psspy.amachint(string=["NUMBER"])
        if ierr > 0:
            print("[ERROR] Failed to retrieve generator bus numbers.")
            return False

        for bus_num in gen_buses[0]:
            # Only modify QT (index 2) and QB (index 3)
            realar_array = [self._f] * 17
            realar_array[2] = 9999.0  # QT (Q max)
            realar_array[3] = -9999.0 # QB (Q min)

            ierr = self.psspy.machine_chng_4(ibus=bus_num, realar=realar_array)
            if ierr > 0:
                print(f"[WARN] Failed to update Q limits at bus {bus_num}.")
        self.grid = GridState()  # TODO Reset state after adding farm but should be a bette way
        self.solve_powerflow()
        self.take_snapshot(timestamp=self.engine.time.start_time)
        return True


    def add_wec_farm(self, farm: WECFarm) -> bool:
        """Add a WEC farm to the PSS®E model.

        This method adds a WEC farm to the PSS®E model by creating the necessary
        electrical infrastructure: a new bus for the WEC farm, a generator on that bus,
        and a transmission line connecting it to the existing grid.

        Args:
            farm (WECFarm): The WEC farm object containing connection details.

        Returns:
            bool: True if the farm is added successfully, False otherwise.

        Raises:
            ValueError: If the WEC farm cannot be added due to invalid parameters.

        Notes:
            The following PSS®E API calls are used:
            
            - ``busdat()``: Get base voltage of connecting bus
                Returns: Base voltage [kV]
            - ``bus_data_4()``: Add new WEC bus (PV type)
                - Base voltage [kV]
            - ``plant_data_4()``: Add plant data to WEC bus
            - ``machine_data_4()``: Add WEC generator to bus
                - PG: Active power generation [MW]
            - ``branch_data_3()``: Add transmission line from WEC bus to grid
                - R: Resistance [pu]
                - X: Reactance [pu]
                - RATEA: Rating A [MVA]
        TODO:
            Fix the hardcoded line R, X, and RATEA values
        """
    
        ierr, rval = self.psspy.busdat(farm.connecting_bus, "BASE")
        
        if ierr > 0:
            print(f"Error retrieving base voltage for bus {farm.connecting_bus}. PSS®E error code: {ierr}")

        # Step 1: Add a new bus
        ierr = self.psspy.bus_data_4(
            ibus=farm.bus_location,
            inode=0, 
            intgar1=2, # Bus type (2 = PV bus)
            realar1=rval, # Base voltage of the from bus in kV
            name=f"WEC BUS {farm.bus_location}",
        )
        if ierr > 0:
            print(f"Error adding bus {farm.bus_location}. PSS®E error code: {ierr}")
            return False

        # Step 2: Add plant data
        ierr = self.psspy.plant_data_4(
            ibus=farm.bus_location,
            inode=0
        )
        if ierr > 0:
            print(f"Error adding plant data to bus {farm.bus_location}. PSS®E error code: {ierr}")
            return False
        
        # Step 3: Add generator
        ierr = self.psspy.machine_data_4(
            ibus=farm.bus_location, 
            id=f"W{farm.id}",
            realar1=0.0, # PG, machine active power (0.0 by default)
        )
        if ierr > 0:
            print(f"Error adding generator {farm.id} to bus {farm.bus_location}. PSS®E error code: {ierr}")
            return False

        # Step 4: Add a branch (line) connecting the existing bus to the new bus
        realar_array = [0.0] * 12
        realar_array[0] = 0.0452  # R
        realar_array[1] = 0.1652  # X
        ratings_array = [0.0] * 12
        ratings_array[0] = 130.00  # RATEA
        ierr = self.psspy.branch_data_3(
            ibus=farm.bus_location,
            jbus=farm.connecting_bus,
            realar=realar_array,
            namear="WEC Line"
        )
        if ierr > 0:
            print(f"Error adding branch from {farm.bus_location} to {farm.connecting_bus}. PSS®E error code: {ierr}")
            return False

        self.grid = GridState()  # TODO: Reset state after adding farm, but should be a better way
        self.solve_powerflow()
        self.take_snapshot(timestamp=self.engine.time.start_time)
        return True
    

    def simulate(self, load_curve: Optional[pd.DataFrame] = None) -> bool:
        """Simulate the PSS®E grid over time with WEC farm updates.
        
        Simulates the PSS®E grid over a series of time snapshots, updating WEC farm 
        generator outputs and optionally bus loads at each time step. For each snapshot,
        the method updates generator power outputs, applies load changes if provided,
        solves the power flow, and captures the grid state.
        
        Args:
            load_curve (Optional[pd.DataFrame]): DataFrame containing load values for 
                each bus at each snapshot. Index should be snapshots, columns should 
                be bus IDs. If None, loads remain constant.

        Returns:
            bool: True if the simulation completes successfully.
            
        Raises:
            Exception: If there is an error setting generator power, setting load data, 
                or solving the power flow at any snapshot.

        Notes:
            The following PSS®E API calls are used for simulation:
            
            - ``machine_chng_4()``: Update WEC generator active power output
                - PG: Active power generation [MW]
            - ``load_data_6()``: Update bus load values (if load_curve provided)
                - P: Active power load [MW]
                - Q: Reactive power load [MVAr]
            - ``fnsl()``: Solve power flow at each time step
        """
        

        for snapshot in tqdm(self.engine.time.snapshots, desc="PSS®E Simulating", unit="step"):
            for farm in self.engine.wec_farms:
                power = farm.power_at_snapshot(snapshot) # pu in MW (1.0 MVA)
                ierr = self.psspy.machine_chng_4(
                        ibus=farm.bus_location, 
                        id=f"W{farm.id}", 
                        realar=[power * farm.BASE] + [self._f]*16) > 0 # no need to use Farm.BASE becuase base was passed in add_wec_farm
                if ierr > 0: 
                    raise Exception(f"Error setting generator power at snapshot {snapshot}")
            if load_curve is not None:
                for bus in load_curve.columns:
                    pl = float(load_curve.loc[snapshot, bus])
                    ierr = self.psspy.load_data_6(
                        ibus=bus, 
                        realar=[pl * self.sbase] + [self._f]*7)
                if ierr > 0:
                    raise Exception(f"Error setting load at bus {bus} on snapshot {snapshot}")
            if self.solve_powerflow():
                self.take_snapshot(timestamp=snapshot)
            else:
                raise Exception(f"Powerflow failed at snapshot {snapshot}")
        return True

    def take_snapshot(self, timestamp: datetime) -> None:
        """Take a snapshot of the current grid state.
        
        Captures the current state of all grid components (buses, generators, lines,
        and loads) at the specified timestamp and updates the grid state object.
        
        Args: 
            timestamp (datetime): The timestamp for the snapshot.

        Returns:
            None
        """
        # --- Append time-series for each component ---
        self.grid.update("bus",    timestamp, self.snapshot_buses())
        self.grid.update("gen",    timestamp, self.snapshot_generators())
        self.grid.update("line", timestamp, self.snapshot_lines())
        self.grid.update("load",   timestamp, self.snapshot_loads())
 
    def snapshot_buses(self) -> pd.DataFrame:
        """Capture current bus state from PSS®E.
        
        Builds a Pandas DataFrame of the current bus state for the loaded PSS®E grid 
        using the PSS®E API. The DataFrame is formatted according to the GridState 
        specification and includes bus voltage, power injection, and load data.
        
        Returns:
            pd.DataFrame: DataFrame with columns: bus, bus_name, type, p, q, v_mag, 
                angle_deg, base. Index represents individual buses.
                
        Raises:
            RuntimeError: If there is an error retrieving bus snapshot data from PSS®E.

        Notes:
            The following PSS®E API calls are used to retrieve bus snapshot data:
            
            Bus Information:
            - ``abuschar()``: Bus names ('NAME')
                Returns: Bus names [string]
            - ``abusint()``: Bus numbers and types ('NUMBER', 'TYPE')  
                Returns: Bus numbers [dimensionless], Bus types [dimensionless]
            - ``abusreal()``: Bus voltages and base kV ('PU', 'ANGLED', 'BASE')
                Returns: Voltage magnitude [pu], Voltage angle [degrees], Base voltage [kV]
            
            Generator Data:
            - ``amachint()``: Generator bus numbers ('NUMBER')
                Returns: Bus numbers [dimensionless]
            - ``amachreal()``: Generator power output ('PGEN', 'QGEN')
                Returns: Active power [MW], Reactive power [MVAr]
            
            Load Data:
            - ``aloadint()``: Load bus numbers ('NUMBER')
                Returns: Bus numbers [dimensionless]
            - ``aloadcplx()``: Load power consumption ('TOTALACT')
                Returns: Complex power [MW + j*MVAr]
        """
        # --- Pull data from PSS®E ---
        ierr1, names = self.psspy.abuschar(string=["NAME"])
        ierr2, ints   = self.psspy.abusint(string=["NUMBER", "TYPE"])
        ierr3, reals  = self.psspy.abusreal(string=["PU", "ANGLED", "BASE"])
        ierr4, gens   = self.psspy.amachint(string=["NUMBER"])
        ierr5, pgen   = self.psspy.amachreal(string=["PGEN"])
        ierr6, qgen   = self.psspy.amachreal(string=["QGEN"])
        ierr7, loads  = self.psspy.aloadint(string=["NUMBER"])
        ierr8, pqload = self.psspy.aloadcplx(string=["TOTALACT"])

        if any(ierr != 0 for ierr in [ierr1, ierr2, ierr3, ierr4, ierr5, ierr6, ierr7, ierr8]):
            raise RuntimeError("Error retrieving bus snapshot data from PSSE.")


        # --- Unpack ---
        bus_numbers, bus_types = ints
        v_mag, angle_deg, base_kv = reals  # base_kv is kV (NOT for p.u. power!)
        gen_bus_ids = gens[0]
        pgen_mw     = pgen[0]
        qgen_mvar   = qgen[0]
        load_bus_ids = loads[0]
        load_cplx    = pqload[0]

        # --- Aggregate gen/load per bus ---
        from collections import defaultdict
        gen_map = defaultdict(lambda: [0.0, 0.0])
        for b, p, q in zip(gen_bus_ids, pgen_mw, qgen_mvar):
            gen_map[b][0] += p
            gen_map[b][1] += q

        load_map = defaultdict(lambda: [0.0, 0.0])
        for b, pq in zip(load_bus_ids, load_cplx):
            load_map[b][0] += pq.real
            load_map[b][1] += pq.imag

        # --- Map type codes ---
        type_map = {3: "Slack", 2: "PV", 1: "PQ"}

        # --- Build rows ---
        rows = []
        for i in range(len(bus_numbers)):
            bus = bus_numbers[i]
            name = names[0][i].strip()
            pgen_b, qgen_b = gen_map[bus]
            pload_b, qload_b = load_map[bus]

            # per-unit on system MVA base
            p_pu = (pgen_b - pload_b) / self.sbase
            q_pu = (qgen_b - qload_b) / self.sbase

            rows.append({
                "bus":       bus, # todo figure out int of str
                "bus_name":  name,
                "type":      type_map.get(bus_types[i], f"Unknown({bus_types[i]})"),
                "p":         p_pu,
                "q":         q_pu,
                "v_mag":     v_mag[i],          # already pu
                "angle_deg": angle_deg[i],      # PSSE returns degrees
                "base":      base_kv[i],        # kV (for info/consistency with PyPSA)
            })

        df = pd.DataFrame(rows)
        df.attrs["df_type"] = "BUS"
        return df

    def snapshot_generators(self) -> pd.DataFrame:
        """Capture current generator state from PSS®E.
        
        Builds a Pandas DataFrame of the current generator state for the loaded PSS®E grid 
        using the PSS®E API. The DataFrame includes generator power output, base MVA, 
        and status information.
        
        Returns:
            pd.DataFrame: DataFrame with columns: gen, bus, p, q, base, status.
                Generator names are formatted as "bus_count" (e.g., "1_1", "1_2").
                
        Raises:
            RuntimeError: If there is an error retrieving generator data from PSS®E.

        Notes:
            The following PSS®E API calls are used to retrieve generator data:
            
            - ``amachint()``: Generator bus numbers and status ('NUMBER', 'STATUS')
                Returns: Bus numbers [dimensionless], Status codes [dimensionless]
            - ``amachreal()``: Generator power and base MVA ('PGEN', 'QGEN', 'MBASE')
                Returns: Active power [MW], Reactive power [MVAr], Base MVA [MVA]
        """
        
        ierr1, int_arr  = self.psspy.amachint(string=["NUMBER", "STATUS"])
        ierr2, real_arr = self.psspy.amachreal(string=["PGEN", "QGEN", "MBASE"])
        if any(ierr != 0 for ierr in [ierr1, ierr2]):
            raise RuntimeError("Error fetching generator (machine) data.")

        bus_ids, statuses = int_arr
        pgen_mw, qgen_mvar, mbases = real_arr

        rows = []
        counter = defaultdict(int)

        for i, bus in enumerate(bus_ids):
            gen_count = counter[bus]
            counter[bus] += 1
            gen_key = f"{bus}_{gen_count+1}"  # always bus number + count

            base = mbases[i] or 100.0
            rows.append({
                "gen":    gen_key,
                "bus":    bus,
                "p":      pgen_mw[i] / base,
                "q":      qgen_mvar[i] / base,
                "base":   base,
                "status": statuses[i],
            })

        df = pd.DataFrame(rows)
        df.attrs["df_type"] = "GEN"
        return df
       
    def snapshot_lines(self) -> pd.DataFrame:
        """Capture current transmission line state from PSS®E.
        
        Builds a Pandas DataFrame of the current transmission line state for the loaded 
        PSS®E grid using the PSS®E API. The DataFrame includes line loading percentages
        and connection information.
        
        Returns:
            pd.DataFrame: DataFrame with columns: line, ibus, jbus, line_pct, status.
                Line names are formatted as "Line_ibus_jbus_count".
                
        Raises:
            RuntimeError: If there is an error retrieving line data from PSS®E.

        Notes:
            The following PSS®E API calls are used to retrieve line data:
            
            - ``abrnchar()``: Line IDs ('ID')
                Returns: Line identifiers [string]
            - ``abrnint()``: Line bus connections and status ('FROMNUMBER', 'TONUMBER', 'STATUS')
                Returns: From bus [dimensionless], To bus [dimensionless], Status [dimensionless]
            - ``abrnreal()``: Line loading percentage ('PCTRATE')
                Returns: Line loading [%]
        """

        ierr1, carray = self.psspy.abrnchar(string=["ID"])
        ids = carray[0]

        ierr2, iarray = self.psspy.abrnint(string=["FROMNUMBER", "TONUMBER", "STATUS"])
        ibuses, jbuses, statuses = iarray

        ierr3, rarray = self.psspy.abrnreal(string=["PCTRATE"])
        pctrates = rarray[0]

        if any(ierr != 0 for ierr in [ierr1, ierr2, ierr3]):
            raise RuntimeError("Error fetching line data from PSSE.")

        rows = []
        counter = defaultdict(int)

        for i in range(len(ibuses)):
            ibus = ibuses[i]
            jbus = jbuses[i]
            counter[(ibus, jbus)] += 1
            idx = counter[(ibus, jbus)]  # starts at 1 now

            line_id = f"Line_{ibus}_{jbus}_{idx}"

            rows.append({
                "line": line_id,
                "ibus": ibus,
                "jbus": jbus,
                "line_pct": pctrates[i],
                "status": statuses[i],
            })

        df = pd.DataFrame(rows)
        df.attrs["df_type"] = "LINE"
        df.index = pd.RangeIndex(start=0, stop=len(df))  # clean index
        return df

    def snapshot_loads(self) -> pd.DataFrame:
        """Capture current load state from PSS®E.
        
        Builds a Pandas DataFrame of the current load state for the loaded PSS®E grid 
        using the PSS®E API. The DataFrame includes load power consumption and status
        information for all buses with loads.
        
        Returns:
            pd.DataFrame: DataFrame with columns: load, bus, p, q, base, status.
                Load names are formatted as "Load_bus_count".
                
        Raises:
            RuntimeError: If there is an error retrieving load data from PSS®E.

        Notes:
            The following PSS®E API calls are used to retrieve load data:
            
            - ``aloadchar()``: Load IDs ('ID')
                Returns: Load identifiers [string]
            - ``aloadint()``: Load bus numbers and status ('NUMBER', 'STATUS')
                Returns: Bus numbers [dimensionless], Status codes [dimensionless]
            - ``aloadcplx()``: Load power consumption ('TOTALACT')
                Returns: Complex power consumption [MW + j*MVAr]
        """
        # --- Load character data: IDs
        ierr1, char_arr = self.psspy.aloadchar(string=["ID"])
        load_ids = char_arr[0]

        # --- Load integer data: bus number and status
        ierr2, int_arr = self.psspy.aloadint(string=["NUMBER", "STATUS"])
        bus_numbers, statuses = int_arr

        # --- Load complex power (in MW/MVAR)
        ierr3, complex_arr = self.psspy.aloadcplx(string=["TOTALACT"])
        total_act = complex_arr[0]

        if any(ierr != 0 for ierr in [ierr1, ierr2, ierr3]):
            raise RuntimeError("Error retrieving load snapshot data from PSSE.")

        rows = []
        counter = defaultdict(int)
        for i in range(len(bus_numbers)):
            bus = bus_numbers[i]
            counter[bus] += 1
            idx = counter[bus]  # starts at 1 now

            rows.append({
                "load":   f"Load_{bus}_{idx}",
                "bus":    bus,
                "p":      total_act[i].real / self.sbase,  # Convert MW to pu
                "q":      total_act[i].imag / self.sbase,  # Convert MVAR to pu
                "base":   self.sbase,
                "status": statuses[i],
            })

        df = pd.DataFrame(rows)
        df.attrs["df_type"] = "LOAD"
        df.index = pd.RangeIndex(start=0, stop=len(df))  # Clean index
        return df
