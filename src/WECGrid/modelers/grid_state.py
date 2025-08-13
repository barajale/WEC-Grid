# src/wecgrid/modelers/grid_state.py

import pandas as pd
from typing import Optional, Dict
from collections import defaultdict


class AttrDict(dict):
    """Dictionary that allows attribute-style access to keys.
    
    This utility class enables accessing dictionary values using dot notation
    (d.key) in addition to the standard bracket notation (d['key']). This is
    used for convenient access to time-series data collections.
    
    Example:
        >>> data = AttrDict({'voltage': df1, 'power': df2})
        >>> data.voltage  # Same as data['voltage']
        >>> data.power = df3  # Same as data['power'] = df3
        
    Raises:
        AttributeError: If the requested attribute/key does not exist.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class GridState:
    """Standardized container for power system snapshot and time-series data.
    
    The GridState class provides a unified data structure for storing power system
    component states across different simulation backends (PSS®E, PyPSA, etc.). It
    maintains both current snapshot data and historical time-series data for buses,
    generators, lines, and loads using standardized DataFrame schemas.
    
    This class enables cross-platform validation and comparison between different
    power system analysis tools by enforcing consistent data formats and units.
    All electrical quantities are stored in per-unit values based on system MVA.
    
    Attributes:
        bus (pd.DataFrame): Current bus state with voltage, power injection data.
        gen (pd.DataFrame): Current generator state with power output data.
        line (pd.DataFrame): Current transmission line state with loading data.
        load (pd.DataFrame): Current load state with power consumption data.
        bus_t (AttrDict): Time-series bus data organized by variable name.
        gen_t (AttrDict): Time-series generator data organized by variable name.
        line_t (AttrDict): Time-series line data organized by variable name.
        load_t (AttrDict): Time-series load data organized by variable name.
        
    Example:
        >>> grid = GridState()
        >>> # Update with current snapshot
        >>> grid.update("bus", timestamp, bus_dataframe)
        >>> # Access current state
        >>> print(f"Number of buses: {len(grid.bus)}")
        >>> # Access time-series data
        >>> voltage_history = grid.bus_t.v_mag  # All bus voltages over time
        
    Notes:
        - All power values are in per-unit on system base MVA
        - Voltage magnitudes are in per-unit, angles in degrees
        - Line loading is expressed as percentage of thermal rating
        - Component IDs must be consistent across all DataFrames
        - Time-series data is automatically maintained when snapshots are updated
        
    DataFrame Schemas:
        Each component DataFrame follows a standardized schema as documented
        in the individual update method and property descriptions.
    """

    def __init__(self):
        """Initialize GridState with empty DataFrames and time-series containers.
        
        Creates empty DataFrame structures for all component types and initializes
        AttrDict containers for time-series data storage. The DataFrames will be
        populated when the first snapshot is taken.
        """
        # Empty DataFrames with appropriate dtypes — index will be set on first update
        empty_df = pd.DataFrame()

        # Snapshot (single-time) dataframes
        self.bus: pd.DataFrame = empty_df.copy()
        self.gen: pd.DataFrame = empty_df.copy()
        self.line: pd.DataFrame = empty_df.copy()
        self.load: pd.DataFrame = empty_df.copy()

        # Time-series dicts (e.g. { "P_MW": DataFrame with rows = time, cols = ID })
        self.bus_t: AttrDict = AttrDict()
        self.gen_t: AttrDict = AttrDict()
        self.line_t: AttrDict = AttrDict()
        self.load_t: AttrDict = AttrDict()

    def __repr__(self) -> str:
        """Return a formatted string representation of the GridState.
        
        Provides a tree-style summary showing the number of components in each
        category and the available time-series variables for each component type.
        
        Returns:
            str: Multi-line string representation showing component counts and
                available time-series data.
                
        Example:
            >>> print(grid)
            GridState:
            ├─ bus:   14
            │   └─ time-series: v_mag, angle_deg, p, q
            ├─ gen:   5
            │   └─ time-series: p, q, status
            ├─ line:  20
            │   └─ time-series: line_pct, status
            └─ load:  11
                └─ time-series: p, q, status
        """
        def ts_keys(d):
            return ", ".join(d.keys()) if d else "none"

        return (
            "GridState:\n"
            f"├─ bus:   {len(self.bus)}\n"
            f"│   └─ time-series: {ts_keys(self.bus_t)}\n"
            f"├─ gen:   {len(self.gen)}\n"
            f"│   └─ time-series: {ts_keys(self.gen_t)}\n"
            f"├─ line:  {len(self.line)}\n"
            f"│   └─ time-series: {ts_keys(self.line_t)}\n"
            f"└─ load:  {len(self.load)}\n"
            f"    └─ time-series: {ts_keys(self.load_t)}"
        )

    def update(self, component: str, timestamp: pd.Timestamp, df: pd.DataFrame):
        """Update snapshot and time-series data for a power system component.
        
        This method updates both the current snapshot DataFrame and the historical
        time-series data for the specified component type. It expects DataFrames
        with standardized schemas and proper df_type attributes.
        
        Args:
            component (str): Component type ("bus", "gen", "line", "load").
            timestamp (pd.Timestamp): Timestamp for this snapshot.
            df (pd.DataFrame): Component data with df.attrs['df_type'] set to
                one of {"BUS", "GEN", "LINE", "LOAD"}.
        
        Raises:
            ValueError: If component is not recognized, df_type is invalid, or
                required ID columns are missing.
                
        DataFrame Schemas:
            
            **Bus DataFrame (df_type="BUS")**:
            
            | Column      | Description                           | Type  | Units    |
            |-------------|---------------------------------------|-------|----------|
            | bus         | Bus number (unique identifier)        | int   | -        |
            | bus_name    | Bus name/label                        | str   | -        |
            | type        | Bus type (Slack/PV/PQ)                | str   | -        |
            | p           | Net active power injection            | float | pu       |
            | q           | Net reactive power injection          | float | pu       |
            | v_mag       | Voltage magnitude                     | float | pu       |
            | angle_deg   | Voltage angle                         | float | degrees  |
            | base        | Base voltage                          | float | kV       |
            
            **Generator DataFrame (df_type="GEN")**:
            
            | Column      | Description                           | Type  | Units    |
            |-------------|---------------------------------------|-------|----------|
            | gen         | Generator ID (e.g., "1_1", "2_1")    | str   | -        |
            | bus         | Connected bus number                  | int   | -        |
            | p           | Active power output                   | float | pu       |
            | q           | Reactive power output                 | float | pu       |
            | base        | Generator base MVA                    | float | MVA      |
            | status      | Generator status (1=online, 0=off)   | int   | -        |
            
            **Line DataFrame (df_type="LINE")**:
            
            | Column      | Description                           | Type  | Units    |
            |-------------|---------------------------------------|-------|----------|
            | line        | Line ID (e.g., "Line_1_2_1")         | str   | -        |
            | ibus        | From bus number                       | int   | -        |
            | jbus        | To bus number                         | int   | -        |
            | line_pct    | Line loading percentage               | float | %        |
            | status      | Line status (1=online, 0=offline)    | int   | -        |
            
            **Load DataFrame (df_type="LOAD")**:
            
            | Column      | Description                           | Type  | Units    |
            |-------------|---------------------------------------|-------|----------|
            | load        | Load ID (e.g., "Load_1_1")           | str   | -        |
            | bus         | Connected bus number                  | int   | -        |
            | p           | Active power consumption              | float | pu       |
            | q           | Reactive power consumption            | float | pu       |
            | base        | System base MVA                       | float | MVA      |
            | status      | Load status (1=connected, 0=off)     | int   | -        |
        
        Notes:
            - All power values (p, q) are in per-unit on the appropriate base
            - Bus types: "Slack" (reference), "PV" (voltage controlled), "PQ" (load)
            - Component IDs must be unique within each component type
            - Line loading is percentage of thermal rating (not per-unit)
            - Status codes: 1 = in-service/online, 0 = out-of-service/offline
            
        Example:
            >>> # Update bus data at current time
            >>> bus_df = create_bus_dataframe()  # With proper schema
            >>> bus_df.attrs['df_type'] = 'BUS'
            >>> grid.update("bus", pd.Timestamp.now(), bus_df)
            >>> 
            >>> # Access updated data
            >>> current_buses = grid.bus
            >>> voltage_timeseries = grid.bus_t.v_mag
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