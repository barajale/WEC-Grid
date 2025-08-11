"""
PyPSA Wrapper Module

This module provides a wrapper class for PyPSA (Python for Power System Analysis) functionality,
specifically designed for Wave Energy Converter (WEC) integration into power systems.

Classes:
    pyPSAWrapper: Main class for managing PyPSA network operations with WEC integration
"""

# Standard Libraries
import os
import sys
from datetime import datetime, timezone, timedelta
import contextlib
import io
import logging
import contextlib

# 3rd Party Libraries
import pypsa
import pandas as pd
import cmath
import numpy as np
import matlab.engine
import pypower.api as pypower
from pandas.tseries.offsets import DateOffset
from math import inf
from typing import Optional, Dict, Any
from collections import defaultdict

from tqdm import tqdm


import grg_pssedata.io as grgio
from grg_pssedata.io import parse_psse_case_file
from typing import Any


from .grid_state import GridState  # used internally

"""
PyPSA implementation of PowerSystemModeler
"""

from .power_system_modeler import PowerSystemModeler
# import PyPSA here
# e.g.: import pypsa

class PyPSAModeler(PowerSystemModeler):
    def __init__(self, engine: Any):
        super().__init__(engine)

        self.network: Optional[pypsa.Network] = None
    

    def __repr__(self) -> str:
        return (
            f"pypsa:\n"
            f"├─ case: {self.engine.case_name}\n"
            f"├─ buses: {len(self.grid.bus)}\n"
            f"├─ generators: {len(self.grid.gen)}\n"
            f"├─ loads: {len(self.grid.load)}\n"
            f"└─ lines: {len(self.grid.line)}"
            f"\n"
            f"Sbase: {self.sbase} MVA"
        )
        

    def init_api(self) -> bool:
        if not self.import_raw_to_pypsa(): return False
        if not self.solve_powerflow(): return False
        self.take_snapshot(timestamp=self.engine.time.start_time)  # populates self.grid
        print("PyPSA software initialized")
        return True
        
    def solve_powerflow(self) -> bool:
        
        # Suppress PyPSA logging
        logger = logging.getLogger("pypsa")
        previous_level = logger.level
        logger.setLevel(logging.WARNING)

        # Optional: suppress stdout too, just in case
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            results = self.network.pf()
        
        # Restore logging level
        logger.setLevel(previous_level)
        # Check convergence
        if not results.converged.all().bool():
            print("[PyPSA WARNING]: Some snapshots failed to converge.")
            failed = results.converged[~results.converged[0]].index.tolist()
            print("Non-converged snapshots:", failed)
            return False
        else:
            return True
        
        
     
    def import_raw_to_pypsa(self) -> bool:
        """
        Builds a PyPSA Network from a parsed PSS/E RAW case file.
        Only sets necessary fields for a valid power flow.
        """
        try:
            # Temporarily silence GRG's print_err
            original_print_err = grgio.print_err
            grgio.print_err = lambda *args, **kwargs: None

            self.parser = parse_psse_case_file(self.engine.case_file)
            
    
            # Restore original print_err
            grgio.print_err = original_print_err

            # Validate case
            if not self.parser or not self.parser.buses:
                print("[GRG ERROR] Parsed case is empty or invalid.")
                return False

            self.sbase = self.parser.sbase
            self.network = pypsa.Network(s_n_mva=self.sbase)

        except Exception as e:
            print(f"[GRG ERROR] Failed to parse case: {e}")
            return False
        
        self.parser.bus_lookup = {bus.i: bus for bus in self.parser.buses}

        # Mapping PSS/E bus types to PyPSA control types
        ide_to_ctrl = {1: "PQ", 2: "PV", 3: "Slack"}

        # --- Add Buses ---
        for bus in self.parser.buses:
            self.network.add("Bus",
                name          = str(bus.i),
                v_nom         = bus.basekv,        # [kV]
                v_mag_pu_set  = bus.vm,             # [pu]
                v_mag_pu_min  = bus.nvlo,           # [pu]
                v_mag_pu_max  = bus.nvhi,           # [pu]
                control       = ide_to_ctrl.get(bus.ide, "PQ"),
            )

        # --- Add Lines (Branches) ---
        for idx, br in enumerate(self.parser.branches):
            line_name = f"L{idx}"
            S_base_MVA = self.parser.sbase
            V_base_kV = self.network.buses.at[str(br.i), "v_nom"]
        
            # Convert PSS®E p.u. values to physical units
            r_ohm = br.r * (V_base_kV ** 2) / S_base_MVA
            x_ohm = br.x * (V_base_kV ** 2) / S_base_MVA
            g_siemens = (br.gi + br.gj) * S_base_MVA / (V_base_kV ** 2)
            b_siemens = (br.bi + br.bj) * S_base_MVA / (V_base_kV ** 2)

            self.network.add("Line",
                name    = line_name,
                bus0    = str(br.i),
                bus1    = str(br.j),
                type    = "",
                r       = r_ohm,
                x       = x_ohm,
                g       = g_siemens,
                b       = b_siemens,
                s_nom   = br.ratea,
                s_nom_extendable = False,
                length  = 0.0,
                v_ang_min = -inf,
                v_ang_max = inf,
            )

            
            
        # --- Add Generators ---
        for idx, g in enumerate(self.parser.generators):
            if g.stat != 1:
                continue
            gname = f"G{idx}"
            S_base_MVA = self.parser.sbase

            # Control type from IDE (bus type), fallback to "PQ"
            ctrl = ide_to_ctrl.get(self.parser.bus_lookup[g.i].ide, "PQ")

            # Active power limits and nominal power
            p_nom = g.pt
            p_nom_min = g.pb
            p_set = g.pg
            p_min_pu = g.pb / g.pt if g.pt != 0 else 0.0  # Avoid div by zero

            # Reactive setpoint
            q_set = g.qg

            # Optional: carrier type (e.g., detect wind)
            carrier = "wind" if getattr(g, "wmod", 0) != 0 else "other"

            self.network.add("Generator",
                name       = gname,
                bus        = str(g.i),
                control    = ctrl,
                p_nom      = p_nom,
                p_nom_extendable = False,
                p_nom_min  = p_nom_min,
                p_nom_max  = p_nom,
                p_min_pu   = p_min_pu,
                p_max_pu   = 1.0,
                p_set      = p_set,
                q_set      = q_set,
                carrier    = carrier,
                efficiency = 1.0,  # Default unless you have a better estimate
            )
            

        # --- Add Loads ---
        for idx, load in enumerate(self.parser.loads):
            if load.status != 1:
                continue  # Skip out-of-service loads

            lname = f"L{idx}"

            self.network.add("Load",
                name   = lname,
                bus    = str(load.i),
                carrier = "AC",        # Default for electrical loads
                p_set  = load.pl,
                q_set  = load.ql,
            )
        # --- Add Transformers ---
        for idx, tx in enumerate(self.parser.transformers):
            p1 = tx.p1
            p2 = tx.p2
            w1 = tx.w1
            w2 = tx.w2

            # Skip transformer if it's out of service (status not equal to 1 = fully in-service)
            if p1.stat != 1:
                continue

            # Transformer name and buses
            name = f"T{idx}"
            bus0 = str(p1.i)
            bus1 = str(p1.j)

            # Apparent power base (MVA)
            s_nom = w1.rata if w1.rata > 0.0 else p2.sbase12

            # Normalize impedance from sbase12 to s_nom
            r = p2.r12 * (p2.sbase12 / s_nom)
            x = p2.x12 * (p2.sbase12 / s_nom)

            # Optional magnetizing admittance (can be set to 0.0 if not used)
            g = p1.mag1 / s_nom if p1.mag1 != 0.0 else 0.0
            b = p1.mag2 / s_nom if p1.mag2 != 0.0 else 0.0

            # Tap ratio and angle shift
            tap_ratio = w1.windv
            phase_shift = w1.ang

            self.network.add("Transformer",
                name        = name,
                bus0        = bus0,
                bus1        = bus1,
                type        = "",         # Use explicit parameters
                model       = "t",        # PyPSA's physical default
                r           = r,
                x           = x,
                g           = g,
                b           = b,
                s_nom       = s_nom,
                s_nom_extendable = False,
                num_parallel = 1,
                tap_ratio   = tap_ratio,
                tap_side    = 0,
                phase_shift = phase_shift,
                #active      = True,
                v_ang_min   = -180,
                v_ang_max   = 180,
            )
           # --- Add Two-Winding Transformers ---     

        # --- Add Shunt Impedances ---
        for idx, sh in enumerate(self.parser.switched_shunts):
            if sh.status != 1:
                continue  # Skip out-of-service shunts

            v_nom = self.network.buses.at[str(sh.i), "v_nom"]  # in kV
            v_sq = v_nom ** 2

            g_siemens = sh.gl / v_sq  # MW / kV^2 → Siemens
            b_siemens = sh.bl / v_sq  # MVAr / kV^2 → Siemens

            shunt_name = f"Shunt_{idx}"

            self.network.add("ShuntImpedance",
                name = shunt_name,
                bus  = str(sh.i),
                g    = g_siemens,
                b    = b_siemens,
                active = True,
            )
        return 1


    def add_wec_farm(self, farm) -> bool:
        try:
            self.network.add("Bus",
                name=str(farm.bus_location),
                v_nom=self.network.buses.at[str(farm.connecting_bus), "v_nom"],
                carrier="AC",
            )
            self.network.add("Line",
                name="WEC Line",
                bus0=str(farm.bus_location),
                bus1=str(farm.connecting_bus),
                r=7.875648,
                x=28.784447,
                s_nom=130.00,
            )
            self.network.add("Generator",
                name=f"W{farm.id}",
                bus=str(farm.bus_location),
                carrier="wave",
                p_set=0.0,
                control="PV",
            )
            self.grid = GridState()  # TODO Reset state after adding farm but should be a bette way
            self.solve_powerflow()
            self.take_snapshot(timestamp=self.engine.time.start_time)  # Update grid state
            return True
        except Exception as e:
            print(f"[PyPSA ERROR]: Failed to add WEC Components: {e}")
            return False  
    
    
    
    def simulate(self, load_curve=None) -> bool:
        # map: bus number (str) -> load name (index)
        bus_to_load = self.network.loads['bus'].astype(str).to_dict()
        inv_map = {v: k for k, v in bus_to_load.items()}  # bus->load
        # (if you prefer your original naming)
        bus_to_load = {str(bus): name for name, bus in self.network.loads['bus'].items()}

        for snapshot in tqdm(self.engine.time.snapshots, desc="PyPSA Simulating", unit="step"):
            # WEC generators
            for farm in self.engine.wec_farms:
                power = farm.power_at_snapshot(snapshot) * farm.BASE  # pu (1.0 MVA ) -> MW 
                # write to the DataFrame, not the Series view
                self.network.generators.at[f"W{farm.id}", "p_set"] = power

            # Loads
            if load_curve is not None:
                for bus in load_curve.columns:
                    load_id = bus_to_load.get(str(bus)) # PU 
                    if load_id is None:
                        continue  # or raise if this should never happen
                    mw = float(load_curve.loc[snapshot, bus]) * self.sbase
                    self.network.loads.at[load_id, "p_set"] = mw

            if self.solve_powerflow():
                self.take_snapshot(timestamp=snapshot)
            else:
                raise Exception(f"Powerflow failed at snapshot {snapshot}")
        return True
    
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """
            TODO fill in later
        """
        self.grid.update("bus",    timestamp, self.snapshot_buses())
        self.grid.update("gen",    timestamp, self.snapshot_generators())
        self.grid.update("line", timestamp, self.snapshot_lines())
        self.grid.update("load",   timestamp, self.snapshot_loads())

    def snapshot_buses(self) -> pd.DataFrame:
        """
        PyPSA bus snapshot with WEC-Grid schema (all powers in per-unit on system S_base).

        Columns:
        bus, bus_name, type, p, q, v_mag, angle_deg, base
        """
        n = self.network
        buses = n.buses  # index = bus names (strings)

        # choose the latest snapshot (or change to a passed-in timestamp)
        if len(n.snapshots) > 0:
            ts = n.snapshots[-1]
            p_MW = getattr(n.buses_t, "p", pd.DataFrame()).reindex(index=[ts], columns=buses.index).iloc[0].fillna(0.0)
            q_MVAr = getattr(n.buses_t, "q", pd.DataFrame()).reindex(index=[ts], columns=buses.index).iloc[0].fillna(0.0)
            vmag_pu = getattr(n.buses_t, "v_mag_pu", pd.DataFrame()).reindex(index=[ts], columns=buses.index).iloc[0].fillna(1.0)
            vang_rad = getattr(n.buses_t, "v_ang", pd.DataFrame()).reindex(index=[ts], columns=buses.index).iloc[0].fillna(0.0)
        else:
            # no time series yet
            idx = buses.index
            p_MW = pd.Series(0.0, index=idx)
            q_MVAr = pd.Series(0.0, index=idx)
            vmag_pu = pd.Series(1.0, index=idx)
            vang_rad = pd.Series(0.0, index=idx)

        df = pd.DataFrame({
            "bus":       buses.index.astype(str),
            "bus_name":  buses.index.astype(str),
            "type":      buses.get("control", pd.Series("PQ", index=buses.index)).fillna("PQ"),
            # per-unit on system base:
            "p":         (p_MW / self.sbase).astype(float),
            "q":         (q_MVAr / self.sbase).astype(float),
            "v_mag":     vmag_pu.astype(float),
            "angle_deg": np.degrees(vang_rad.astype(float)),
            # base kV (like your PSSE column):
            "base":      buses.get("v_nom", pd.Series(np.nan, index=buses.index)).astype(float),
        })

        df.attrs["df_type"] = "BUS"
        return df
            

    def snapshot_generators(self) -> pd.DataFrame:
        """
        PyPSA generator snapshot with WEC-Grid schema (powers in p.u. on system S_base).
        Generator names follow "<bus>_<counter>" convention like PSS®E.
        """
        n = self.network
        gens = n.generators
        sbase = self.sbase

        if len(n.snapshots) > 0:
            ts = n.snapshots[-1]
            p_MW  = getattr(n.generators_t, "p",      pd.DataFrame()).reindex(index=[ts], columns=gens.index).iloc[0].fillna(0.0)
            q_MVAr= getattr(n.generators_t, "q",      pd.DataFrame()).reindex(index=[ts], columns=gens.index).iloc[0].fillna(0.0)
            stat  = getattr(n.generators_t, "status", pd.DataFrame()).reindex(index=[ts], columns=gens.index).iloc[0]
            if stat.isna().all() and "active" in gens.columns:
                stat = gens["active"].astype(int).reindex(gens.index).fillna(1)
            else:
                stat = stat.fillna(1).astype(int)
        else:
            idx = gens.index
            p_MW  = pd.Series(0.0, index=idx)
            q_MVAr= pd.Series(0.0, index=idx)
            stat  = pd.Series(1,   index=idx, dtype=int)

        # Counter per bus for naming
        counter = defaultdict(int)
        bus_nums = []
        gen_names = []

        for bus in gens["bus"]:
            try:
                bus_num = int(bus)
            except Exception:
                bus_num = bus
            bus_nums.append(bus_num)
            gen_names.append(f"{bus_num}_{counter[bus_num] + 1}")
            counter[bus_num] += 1

        df = pd.DataFrame({
            "gen":    gen_names,
            "bus":    bus_nums,
            "p":      (p_MW   / sbase).astype(float),
            "q":      (q_MVAr / sbase).astype(float),
            "base":   float(sbase),
            "status": stat.astype(int),
        })

        df.attrs["df_type"] = "GEN"
        return df
    

    def snapshot_lines(self) -> pd.DataFrame:
        """
        Snapshot of transmission lines (branches) in WEC-Grid schema.
        Columns: line, ibus, jbus, line_pct, status
        - line_pct is computed from apparent power flow vs s_nom at latest snapshot.
        """
        n = self.network

        # choose latest snapshot if available
        if len(n.snapshots) > 0:
            ts = n.snapshots[-1]
            p0 = getattr(n.lines_t, "p0", pd.DataFrame()).reindex(index=[ts], columns=n.lines.index).iloc[0].fillna(0.0)
            q0 = getattr(n.lines_t, "q0", pd.DataFrame()).reindex(index=[ts], columns=n.lines.index).iloc[0].fillna(0.0)
            p1 = getattr(n.lines_t, "p1", pd.DataFrame()).reindex(index=[ts], columns=n.lines.index).iloc[0].fillna(0.0)
            q1 = getattr(n.lines_t, "q1", pd.DataFrame()).reindex(index=[ts], columns=n.lines.index).iloc[0].fillna(0.0)
        else:
            # no time series → assume zero flow
            idx = n.lines.index
            p0 = pd.Series(0.0, index=idx)
            q0 = pd.Series(0.0, index=idx)
            p1 = pd.Series(0.0, index=idx)
            q1 = pd.Series(0.0, index=idx)

        rows = []
        counter = defaultdict(int)

        for line_name, line in n.lines.iterrows():
            ibus_name, jbus_name = line.bus0, line.bus1

            # Try to preserve numeric bus IDs (we named PyPSA buses as strings of PSS/E numbers)
            def _busnum(name: str) -> int:
                try:
                    return int(name)
                except Exception:
                    # fallback: 1-based position
                    return n.buses.index.get_loc(name) + 1

            ibus = _busnum(ibus_name)
            jbus = _busnum(jbus_name)

            counter[(ibus, jbus)] += 1
            k = counter[(ibus, jbus)]  # start at 1 now
            line_id = f"Line_{ibus}_{jbus}_{k}"

            # apparent power (MVA) at each end
            S0 = np.hypot(p0[line_name], q0[line_name])
            S1 = np.hypot(p1[line_name], q1[line_name])
            Smax = max(S0, S1)

            s_nom = float(line.s_nom) if pd.notna(line.s_nom) else np.nan
            line_pct = float(100.0 * Smax / s_nom) if s_nom and s_nom > 0 else np.nan

            rows.append({
                "line":     line_id,
                "ibus":     ibus,
                "jbus":     jbus,
                "line_pct": line_pct,  # % of s_nom at latest snapshot
                "status":   1  # hard coded
            })

        df = pd.DataFrame(rows)
        df.attrs["df_type"] = "LINE"
        df.index = pd.RangeIndex(len(df))
        return df


    def snapshot_loads(self) -> pd.DataFrame:
        """
        Snapshot of all loads (PyPSA) in WEC-Grid schema.

        Columns: load, bus, p, q, base, status
        - p, q are per-unit on system base self.sbase
        - load name: 'Load_{bus}_{idx}' with per-bus counter
        """
        n = self.network
        sbase = float(self.sbase)

        # latest snapshot values (MW / MVAr)
        if len(n.snapshots) and hasattr(n.loads_t, "p") and hasattr(n.loads_t, "q"):
            ts = n.snapshots[-1]
            p_MW = n.loads_t.p.reindex(index=[ts], columns=n.loads.index).iloc[0].fillna(0.0)
            q_MVAr = n.loads_t.q.reindex(index=[ts], columns=n.loads.index).iloc[0].fillna(0.0)
        else:
            idx = n.loads.index
            p_MW = pd.Series(0.0, index=idx)
            q_MVAr = pd.Series(0.0, index=idx)

        # helper: recover numeric bus id; PyPSA bus names are strings
        def _busnum(name: str) -> int:
            try:
                return int(name)
            except Exception:
                return n.buses.index.get_loc(name) + 1  # stable fallback

        # status: use 'active' if present, else assume in-service
        has_active = "active" in getattr(n.loads, "columns", [])
        status_series = (n.loads["active"].astype(bool) if has_active
                        else pd.Series(True, index=n.loads.index))

        rows, counters = [], defaultdict(int)
        for load_name, rec in n.loads.iterrows():
            bus = _busnum(rec.bus)
            counters[bus] += 1
            k = counters[bus]  # start at 1 now

            rows.append({
                "load":   f"Load_{bus}_{k}",
                "bus":    bus,
                "p":      float(p_MW.get(load_name, 0.0)) / sbase,
                "q":      float(q_MVAr.get(load_name, 0.0)) / sbase,
                "base":   sbase,
                "status": 1 if bool(status_series.get(load_name, True)) else 0,
            })

        df = pd.DataFrame(rows)
        df.attrs["df_type"] = "LOAD"
        df.index = pd.RangeIndex(len(df))
        return df


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # implement additional methods, e.g.
    # def solve_powerflow(self) -> None: ...
    # def get_bus_dataframe_t(self) -> pandas.DataFrame: ...

# Local Libraries (updated with relative imports)
#from ..utilities.util import read_paths  # Relative import for utilities/util.py
#from ..plot.pypsa_viz import PyPSAVisualizer

# # Initialize the PATHS dictionary
# PATHS = read_paths()
# CURR_DIR = os.path.dirname(os.path.abspath(__file__))


# class PyPSAModeler:
#     """
#     A wrapper class for PyPSA network operations with WEC integration.

#     This class provides methods for initializing, modifying, and analyzing power systems
#     with integrated Wave Energy Converters using PyPSA.

#     Attributes:
#         case_file (str): Path to the case file
#         pypsa_history (dict): History of PyPSA operations
#         pypsa_object_history (dict): History of PyPSA objects
#         dataframe (pd.DataFrame): Current network state
#         flow_data (dict): Power flow data
#         WecGridCore (object): Parent WecGridCore reference
#         timestamp_start (datetime): Initialization timestamp
#     """

#     def __init__(self, case : str, engine: "WecGridEngine"):
#         """
#         Initialize PyPSA wrapper.

#         Args:
#             case (str): Path to case file
#             WecGridCore (object): Reference to parent WecGridCore object

#         Returns:
#             None
#         """
#         self.case_file = case
#         self.engine = engine
        
#         #Grid dfs
#         self.bus_dataframe = pd.DataFrame()
#         self.generator_dataframe = pd.DataFrame()
#         self.branches_dataframe = pd.DataFrame()
#         self.loads_dataframe = pd.DataFrame()
#         self.two_winding_dataframe = pd.DataFrame()
#         self.three_winding_dataframe = pd.DataFrame()
        
        
#         # Wrapper 
#         self.snapshots = engine.snapshots
#         self.snapshot_history = []
#         self.load_profiles = pd.DataFrame()
#         self.viz = PyPSAVisualizer(self)
#         self.parser = None
        
        
        
#         # API objects
#         self.network = None
        
#     def init_api(self)-> bool:
#         if self.import_raw_to_pypsa():
#             if self.pf():
#                 print("PyPSA software initialized")
#                 return True
#         else:
#             print("Failed to initialize pyPSA network.")
#             return False
     
#     def import_raw_to_pypsa(self) -> bool:
#         """
#         Builds a PyPSA Network from a parsed PSS/E RAW case file.
#         Only sets necessary fields for a valid power flow.
#         """
#         try:
#             # Temporarily silence GRG's print_err
#             original_print_err = grgio.print_err
#             grgio.print_err = lambda *args, **kwargs: None

#             self.parser = parse_psse_case_file(self.case_file)
            
    
#             # Restore original print_err
#             grgio.print_err = original_print_err

#             # Validate case
#             if not self.parser or not self.parser.buses:
#                 print("[GRG ERROR] Parsed case is empty or invalid.")
#                 return False

#             self.network = pypsa.Network(s_n_mva=self.parser.sbase)

#         except Exception as e:
#             print(f"[GRG ERROR] Failed to parse case: {e}")
#             return False
        
#         self.parser.bus_lookup = {bus.i: bus for bus in self.parser.buses}

#         # Mapping PSS/E bus types to PyPSA control types
#         ide_to_ctrl = {1: "PQ", 2: "PV", 3: "Slack"}

#         # --- Add Buses ---
#         for bus in self.parser.buses:
#             self.network.add("Bus",
#                 name          = str(bus.i),
#                 v_nom         = bus.basekv,        # [kV]
#                 v_mag_pu_set  = bus.vm,             # [pu]
#                 v_mag_pu_min  = bus.nvlo,           # [pu]
#                 v_mag_pu_max  = bus.nvhi,           # [pu]
#                 control       = ide_to_ctrl.get(bus.ide, "PQ"),
#             )

#         # --- Add Lines (Branches) ---
#         for idx, br in enumerate(self.parser.branches):
#             line_name = f"L{idx}"
#             S_base_MVA = self.parser.sbase
#             V_base_kV = self.network.buses.at[str(br.i), "v_nom"]
        
#             # Convert PSS®E p.u. values to physical units
#             r_ohm = br.r * (V_base_kV ** 2) / S_base_MVA
#             x_ohm = br.x * (V_base_kV ** 2) / S_base_MVA
#             g_siemens = (br.gi + br.gj) * S_base_MVA / (V_base_kV ** 2)
#             b_siemens = (br.bi + br.bj) * S_base_MVA / (V_base_kV ** 2)

#             self.network.add("Line",
#                 name    = line_name,
#                 bus0    = str(br.i),
#                 bus1    = str(br.j),
#                 type    = "",
#                 r       = r_ohm,
#                 x       = x_ohm,
#                 g       = g_siemens,
#                 b       = b_siemens,
#                 s_nom   = br.ratea,
#                 s_nom_extendable = False,
#                 length  = 0.0,
#                 v_ang_min = -inf,
#                 v_ang_max = inf,
#             )
#             # print(f"Line {line_name}:")
#             # print(f" bus0 : {str(br.i)} - bus1: {str(br.j)}")
#             # print(f"  Original values (p.u.): r={br.r}, x={br.x}, g={br.gi + br.gj}, b={br.bi + br.bj}")
#             # print(f"  Converted values (physical units): r_ohm={r_ohm:.6f}, x_ohm={x_ohm:.6f}, g_siemens={g_siemens:.6f}, b_siemens={b_siemens:.6f}")
            
            
#         # --- Add Generators ---
#         for idx, g in enumerate(self.parser.generators):
#             if g.stat != 1:
#                 continue
#             gname = f"G{idx}"
#             S_base_MVA = self.parser.sbase

#             # Control type from IDE (bus type), fallback to "PQ"
#             ctrl = ide_to_ctrl.get(self.parser.bus_lookup[g.i].ide, "PQ")

#             # Active power limits and nominal power
#             p_nom = g.pt
#             p_nom_min = g.pb
#             p_set = g.pg
#             p_min_pu = g.pb / g.pt if g.pt != 0 else 0.0  # Avoid div by zero

#             # Reactive setpoint
#             q_set = g.qg

#             # Optional: carrier type (e.g., detect wind)
#             carrier = "wind" if getattr(g, "wmod", 0) != 0 else "other"

#             self.network.add("Generator",
#                 name       = gname,
#                 bus        = str(g.i),
#                 control    = ctrl,
#                 p_nom      = p_nom,
#                 p_nom_extendable = False,
#                 p_nom_min  = p_nom_min,
#                 p_nom_max  = p_nom,
#                 p_min_pu   = p_min_pu,
#                 p_max_pu   = 1.0,
#                 p_set      = p_set,
#                 q_set      = q_set,
#                 carrier    = carrier,
#                 #active     = in_service,
#                 efficiency = 1.0,  # Default unless you have a better estimate
#             )
            

#         # --- Add Loads ---
#         for idx, load in enumerate(self.parser.loads):
#             if load.status != 1:
#                 continue  # Skip out-of-service loads

#             lname = f"L{idx}"

#             self.network.add("Load",
#                 name   = lname,
#                 bus    = str(load.i),
#                 carrier = "AC",        # Default for electrical loads
#                 p_set  = load.pl,
#                 q_set  = load.ql,
#             )
#         # --- Add Transformers ---
#         for idx, tx in enumerate(self.parser.transformers):
#             p1 = tx.p1
#             p2 = tx.p2
#             w1 = tx.w1
#             w2 = tx.w2

#             # Skip transformer if it's out of service (status not equal to 1 = fully in-service)
#             if p1.stat != 1:
#                 continue

#             # Transformer name and buses
#             name = f"T{idx}"
#             bus0 = str(p1.i)
#             bus1 = str(p1.j)

#             # Apparent power base (MVA)
#             s_nom = w1.rata if w1.rata > 0.0 else p2.sbase12

#             # Normalize impedance from sbase12 to s_nom
#             r = p2.r12 * (p2.sbase12 / s_nom)
#             x = p2.x12 * (p2.sbase12 / s_nom)

#             # Optional magnetizing admittance (can be set to 0.0 if not used)
#             g = p1.mag1 / s_nom if p1.mag1 != 0.0 else 0.0
#             b = p1.mag2 / s_nom if p1.mag2 != 0.0 else 0.0

#             # Tap ratio and angle shift
#             tap_ratio = w1.windv
#             phase_shift = w1.ang

#             self.network.add("Transformer",
#                 name        = name,
#                 bus0        = bus0,
#                 bus1        = bus1,
#                 type        = "",         # Use explicit parameters
#                 model       = "t",        # PyPSA's physical default
#                 r           = r,
#                 x           = x,
#                 g           = g,
#                 b           = b,
#                 s_nom       = s_nom,
#                 s_nom_extendable = False,
#                 num_parallel = 1,
#                 tap_ratio   = tap_ratio,
#                 tap_side    = 0,
#                 phase_shift = phase_shift,
#                 #active      = True,
#                 v_ang_min   = -180,
#                 v_ang_max   = 180,
#             )
#            # --- Add Two-Winding Transformers ---     

#         # --- Add Shunt Impedances ---
#         for idx, sh in enumerate(self.parser.switched_shunts):
#             if sh.status != 1:
#                 continue  # Skip out-of-service shunts

#             v_nom = self.network.buses.at[str(sh.i), "v_nom"]  # in kV
#             v_sq = v_nom ** 2

#             g_siemens = sh.gl / v_sq  # MW / kV^2 → Siemens
#             b_siemens = sh.bl / v_sq  # MVAr / kV^2 → Siemens

#             shunt_name = f"Shunt_{idx}"

#             self.network.add("ShuntImpedance",
#                 name = shunt_name,
#                 bus  = str(sh.i),
#                 g    = g_siemens,
#                 b    = b_siemens,
#                 active = True,
#             )
#         return 1

#     def pf(self) -> bool:
#         """
#         Runs power flow silently and checks for convergence.
#         """
#         try:
#             # Suppress PyPSA logging
#             logger = logging.getLogger("pypsa")
#             previous_level = logger.level
#             logger.setLevel(logging.WARNING)

#             # Optional: suppress stdout too, just in case
#             with io.StringIO() as buf, contextlib.redirect_stdout(buf):
#                 results = self.network.pf()

#             # Restore logging level
#             logger.setLevel(previous_level)

#         except Exception as e:
#             print("[PyPSA ERROR]:", str(e))
#             return False

#         # Check convergence
#         if not results.converged.all().bool():
#             print("[PyPSA WARNING]: Some snapshots failed to converge.")
#             failed = results.converged[~results.converged[0]].index.tolist()
#             print("Non-converged snapshots:", failed)
#             return False

#         # Update internal dataframes
#         self.bus_dataframe = self.network.buses.copy()
#         self.generator_dataframe = self.network.generators.copy()
#         self.branches_dataframe = self.network.lines.copy()
#         return True
        
#     def add_wec(self, model: str, ibus: int, jbus: int) -> bool:
#         try:
#             self.network.add("Bus",
#                 name=str(ibus),
#                 v_nom=self.network.buses.at[str(jbus), "v_nom"],
#                 carrier="AC",
#             )
#             self.network.add("Line",
#                 name="WEC Line",
#                 bus0=str(ibus),
#                 bus1=str(jbus),
#                 r=7.875648,
#                 x=28.784447,
#                 s_nom=130.00,
#             )
#             for i, wec in enumerate(self.engine.wecObj_list):
#                 wec.gen_id = f"W{i}"
#                 wec.gen_name = f"{wec.gen_id}-{wec.model}-{wec.ID}"
#                 self.network.add("Generator",
#                     name=str(wec.gen_id),
#                     bus=str(ibus),
#                     carrier="wave",
#                     p_nom=0.03,
#                     p_nom_max = 0.03,
#                     p_max_pu = 1.0,
#                     p_set=0.001,
#                     control="PV",
#                 )
#             self.pf()
#             return True
#         except Exception as e:
#             print(f"[PyPSA ERROR]: Failed to add WEC Components: {e}")
#             return False  

    # def simulate(self, snapshots=None, sim_length=None, load_curve=False, plot=True)->bool:

    #     p_set_data = {}
    #     # Iterate over all WEC objects to gather data
    #     for idx, wec in enumerate(self.engine.wecObj_list):
    #         '''
    #         This is assuming the WEC output is in kW, so we convert it to MW. 
    #         0.016 from wec-sim is = 16kW, so we divide by 1000 to get MW.
    #         '''
    #         pg_data = wec.dataframe["pg"]
    #         p_set_data[wec.gen_id] = pg_data.to_list()
        
        
    #     self.network.set_snapshots(self.snapshots) 
    #     self.network.generators_t.p_set = pd.DataFrame(p_set_data, index=self.snapshots)
    #     if load_curve:
    #         if self.load_profiles is None or self.load_profiles.empty:
    #             self.engine.generate_load_profiles()
    #         self.network.loads_t.p_set = self.load_profiles.copy()
        
    #     self.pf()
        
    #     if plot:
    #         self.viz.plot_all()
    #     return True