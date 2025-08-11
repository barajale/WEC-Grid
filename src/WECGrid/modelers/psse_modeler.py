"""
PSS®E Modeler - Barebones Implementation
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
    new_target = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = new_target
    try:
        yield
    finally:
        sys.stdout = old_stdout
        
class PSSEModeler(PowerSystemModeler):
    def __init__(self, engine: Any):
        super().__init__(engine)

    def __repr__(self) -> str:
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
        """Initialize the PSS®E environment and load the case."""
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
        """Run powerflow and update self.grid."""
        ierr = self.psspy.fnsl()
        ival = self.psspy.solved()
        if ierr != 0 or ival != 0:
            print(f"[ERROR] Powerflow not solved. PSS®E error code: {ierr}, Solved Status: {ival}")
            #TODO error handling here 
            return False
        return True
        #TODO not sure if I should be calling take_snapshot here or in simulate? maybe both?

    def adjust_reactive_lim(self) -> bool:
        """
        Adjusts all generators in the PSSE case to remove reactive power limits
        by setting QT = +9999 and QB = -9999.
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

    #def add_wec_farm(self, model: str, from_bus: int, to_bus: int) -> bool:
    def add_wec_farm(self, farm: WECFarm) -> bool:
        """Inject a WEC farm into the PSS®E model.

        Adds a WEC farm to the PSSE model by:
        1. Adding a new bus.
        2. Adding a generator to the bus to represent the farm.
        3. Adding a branch (line) connecting the new bus to an existing bus.

        Parameters:
        - model (str): Model identifier for the WEC system.
        - ID (int): Unique identifier for the WEC system.
        - ibus (int): New bus ID for the WEC system. (WEC BUS)
        - jbus (int): Existing bus ID to connect the line from.
        """
    
        ierr, rval = self.psspy.busdat(farm.connecting_bus, "BASE")
        
        if ierr > 0:
            print(f"Error retrieving base voltage for bus {farm.connecting_bus}. PSS®E error code: {ierr}")

        # Step 1: Add a new bus
        
        
        
        ierr = self.psspy.bus_data_4(
            ibus=farm.bus_location,
            inode=0, 
            intgar1=2, # Bus type (2 = PV bus ), area, zone, owner
            realar1=rval, # Base voltage of the from bus in kV
            name= f"WEC BUS {farm.bus_location}", # Name of the bus
        )
        if ierr > 0:
            print(f"Error adding bus {farm.bus_location}. PSS®E error code: {ierr}")
            return False

        # Step 2: Add plant data
        ierr = self.psspy.plant_data_4(
            ibus=farm.bus_location, # add plant a wec bus
            inode=0
        )
        if ierr > 0:
            print(f"Error adding plant data to bus {farm.bus_location}. PSS®E error code: {ierr}")
            return False
        
        
        ierr = self.psspy.machine_data_4(
            ibus=farm.bus_location, 
            id = farm.gen_id, # maybe should just be a number? come back
            realar1= 0.0, # PG, machine active power (0.0 by default)
        )

        if ierr > 0:
            print(
                f"Error adding generator {farm.gen_id} to bus {farm.bus_location}. PSS®E error code: {ierr}"
            )
            return False

        # Step 4: Add a branch (line) connecting the existing bus to the new bus
        realar_array = [0.0] * 12
        realar_array[0] = 0.0452  # R
        realar_array[1] = 0.1652  # X
        ratings_array = [0.0] * 12
        ratings_array[0] = 130.00  # RATEA
        ierr = self.psspy.branch_data_3(
            ibus=farm.bus_location, # from bus
            jbus=farm.connecting_bus,  # to bus
            realar=realar_array,
            namear="WEC Line"
        )
        if ierr > 0:
            print(
                f"Error adding branch from {farm.bus_location} to {farm.connecting_bus}. PSS®E error code: {ierr}"
            )
            return False

        self.grid = GridState()  # TODO Reset state after adding farm but should be a bette way
        self.solve_powerflow()
        self.take_snapshot(timestamp=self.engine.time.start_time)
        return True
    
    

    def simulate(self, load_curve: Optional[pd.DataFrame] = None) -> bool:
        """Run a time-series simulation with WEC data injection."""
        

        for snapshot in tqdm(self.engine.time.snapshots, desc="PSS®E Simulating", unit="step"):
            for farm in self.engine.wec_farms:
                power = farm.power_at_snapshot(snapshot) # pu in MW (1.0 MVA)
                ierr = self.psspy.machine_chng_4(
                        ibus=farm.bus_location, 
                        id=farm.gen_id, 
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
        """
            TODO fill in later
        """
        # --- Append time-series for each component ---
        self.grid.update("bus",    timestamp, self.snapshot_buses())
        self.grid.update("gen",    timestamp, self.snapshot_generators())
        self.grid.update("line", timestamp, self.snapshot_lines())
        self.grid.update("load",   timestamp, self.snapshot_loads())


    
    def snapshot_buses(self) -> pd.DataFrame:
        """
        Returns a snapshot DataFrame of all buses following GridState standard.
        Includes: bus, bus_name, type, p, q, v_mag, angle_deg, base, status

        abuschar()
            'NAME' Bus name (12 characters)
          
        abusint()  
            'NUMBER' Bus number
            'TYPE' Bus type code

        abusreal()
            'BASE' Bus base voltage, in kV
            'PU' Actual bus voltage magnitude, in pu
            'ANGLE' Bus voltage phase angle, in radians
            
        amachint() - Bus number and type
            'NUMBER' Bus number
        
        amachreal()
            'PGEN' Active power output, in MW
            'QGEN' Reactive power output, in Mvar

        aloadint()
            'NUMBER' Bus number
        
        aloadcplx()
            'TOTALACT' Actual in-service load (in MW and Mvar)

        """
        # --- Pull data from PSS®E ---
        ierr1, names = self.psspy.abuschar(string=["NAME"])
        ierr2, ints   = self.psspy.abusint(string=["NUMBER", "TYPE"])
        ierr3, reals  = self.psspy.abusreal(string=["PU", "ANGLE", "BASE"])
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
                "bus":       bus,
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
        """Snapshot of transmission lines (branches) in standardized format."""

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
        """
        Snapshot of all in-service loads in standardized format.

        Returns:
            DataFrame with columns: load, bus, p, q, base, status
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
