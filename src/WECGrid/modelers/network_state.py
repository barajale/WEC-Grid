# src/wecgrid/modelers/network_state.py

import pandas as pd
from typing import Optional, Dict
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
    Standardized container for power system snapshot and time-series data.

    Attributes:
        - .bus, .gen, .branch, .load → current snapshots as DataFrames
        - .bus_t, .gen_t, .branch_t, .load_t → time-series dicts (AttrDict[str → DataFrame])
    """

    def __init__(self):
        # Snapshot (single-time) dataframes
        self.bus: Optional[pd.DataFrame] = None
        self.gen: Optional[pd.DataFrame] = None
        self.line: Optional[pd.DataFrame] = None
        self.load: Optional[pd.DataFrame] = None

        # Time-series dicts (e.g. { "P_MW": DataFrame with rows = time, cols = ID })
        self.bus_t: AttrDict = AttrDict()
        self.gen_t: AttrDict = AttrDict()
        self.line_t: AttrDict = AttrDict()
        self.load_t: AttrDict = AttrDict()

    def __repr__(self) -> str:
        def ts_keys(d):
            return ", ".join(d.keys()) if d else "none"

        return (
            "NetworkState:\n"
            f"├─ bus:      {len(self.bus) if self.bus is not None else 0}\n"
            f"│   └─ time-series: {ts_keys(self.bus_t)}\n"
            f"├─ gen:      {len(self.gen) if self.gen is not None else 0}\n"
            f"│   └─ time-series: {ts_keys(self.gen_t)}\n"
            f"├─ line:   {len(self.line) if self.line is not None else 0}\n"
            f"│   └─ time-series: {ts_keys(self.line_t)}\n"
            f"└─ load:     {len(self.load) if self.load is not None else 0}\n"
            f"    └─ time-series: {ts_keys(self.load_t)}"
        )

    def update(self, component: str, timestamp: pd.Timestamp, df: pd.DataFrame):
        """
        Update snapshot and time-series for a component ("bus", "gen", "line", "load").
        Expects df.attrs['df_type'] in {"BUS","GEN","LINE","LOAD"}.
        """

        if df is None or df.empty:
            return

        # --- figure out the ID column for this df_type ---
        df_type = df.attrs.get("df_type", None)
        id_map = {"BUS": "bus", "GEN": "gen", "LINE": "line", "LOAD": "load"}
        id_col = id_map.get(df_type)
        if id_col is None:
            raise ValueError(f"Cannot determine ID column from df_type='{df_type}'")

        # --- ensure the ID is a real column and set as the index for alignment ---
        if id_col in df.columns:
            pass
        elif df.index.name == id_col:
            df = df.reset_index()
        else:
            raise ValueError(f"'{id_col}' not found in columns or as index for df_type='{df_type}'")

        df = df.copy()
        df.set_index(id_col, inplace=True)   # now index = IDs (bus #, gen ID, etc.)

        # keep snapshot (indexed by ID)
        if not hasattr(self, component):
            raise ValueError(f"No snapshot attribute for component '{component}'")
        setattr(self, component, df)

        # --- write into the time-series store ---
        t_attr = getattr(self, f"{component}_t", None)
        if t_attr is None:
            raise ValueError(f"No time-series attribute for component '{component}'")

        # for each measured variable, maintain a DataFrame with:
        #   rows    = timestamps
        #   columns = IDs (df.index)
        for var in df.columns:
            series = df[var]  # index = IDs, values = this variable for this snapshot

            if var not in t_attr:
                t_attr[var] = pd.DataFrame()

            tdf = t_attr[var]
            # add any new IDs as columns
            missing = series.index.difference(tdf.columns)
            if len(missing) > 0:
                tdf[missing] = pd.NA

            # set the row for this timestamp, aligned by ID
            tdf.loc[timestamp, series.index] = series.values
            t_attr[var] = tdf