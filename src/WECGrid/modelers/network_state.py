# src/wecgrid/modelers/network_state.py

import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

class AttrDict(dict):
    """Dictionary that allows attribute-style access: d.key == d['key']"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value
        

class NetworkState:
    """
    Standardized snapshot and time-series container for grid component data.
    Each attribute (e.g., bus, gen) holds the latest snapshot as a DataFrame.
    *_t attributes hold a dictionary of DataFrames indexed by time.
    """

    def __init__(self):
        # Current snapshot DataFrames
        self.bus: Optional[pd.DataFrame]    = None
        self.gen: Optional[pd.DataFrame]    = None
        self.branch: Optional[pd.DataFrame] = None
        self.load: Optional[pd.DataFrame]   = None

        # Time-series dictionaries (e.g., { "PGEN_MW": DataFrame })
        self.bus_t    = AttrDict()
        self.gen_t    = AttrDict()
        self.branch_t = AttrDict()
        self.load_t   = AttrDict()
        
        
    def __repr__(self) -> str:
        def ts_keys(d):
            return ", ".join(d.keys()) if d else "none"

        return (
            "NetworkState:\n"
            f"├─ bus:      {len(self.bus) if self.bus is not None else 0}\n"
            f"│   └─ time-series: {ts_keys(self.bus_t)}\n"
            f"├─ gen:      {len(self.gen) if self.gen is not None else 0}\n"
            f"│   └─ time-series: {ts_keys(self.gen_t)}\n"
            f"├─ branch:   {len(self.branch) if self.branch is not None else 0}\n"
            f"│   └─ time-series: {ts_keys(self.branch_t)}\n"
            f"└─ load:     {len(self.load) if self.load is not None else 0}\n"
            f"    └─ time-series: {ts_keys(self.load_t)}"
        )

    def append_snapshot(self, component: str, timestamp: pd.Timestamp, df: pd.DataFrame):
        """
        Add a snapshot for a component at a given timestamp.
        Uses df.attrs["df_type"] to determine which ID column to use.
        """
        if df is None or df.empty:
            return

        # Determine ID column based on df_type
        df_type = df.attrs.get("df_type", None)

        id_map = {
            "BUS": "BUS_ID",
            "GEN": "GEN_ID",
            "BRANCH": "BRANCH_NAME",
            "LOAD": "BUS_NUMBER",
        }

        id_col = id_map.get(df_type, None)

        if id_col is None or id_col not in df.columns:
            raise ValueError(f"Cannot determine ID column from df_type='{df_type}'")

        df = df.copy()
        df.set_index(id_col, inplace=True, drop=True)
        df = df.sort_index(axis=1)

        t_attr = getattr(self, f"{component}_t", None)
        if t_attr is None:
            raise ValueError(f"No _t attribute for component '{component}'")

        for col in df.columns:
            series = df[col]

            if col not in t_attr:
                t_attr[col] = pd.DataFrame(columns=series.index)

            t_attr[col].loc[timestamp] = series