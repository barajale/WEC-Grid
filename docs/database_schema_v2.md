# WEC-Grid Database Schema v2.0
## Multi-Software Backend Support

### Core Simulation Tables

```sql
-- Main simulation metadata (shared across all software)
CREATE TABLE grid_simulations (
    grid_sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_name TEXT NOT NULL,           -- e.g., "IEEE_14_bus"
    software TEXT NOT NULL,            -- "PSSE", "PyPSA", "GridLAB-D", etc.
    sbase_mva REAL NOT NULL,          -- System base MVA (e.g., 100)
    timestamp_start TEXT NOT NULL,     -- ISO format timestamp
    timestamp_end TEXT,                -- ISO format timestamp (NULL if ongoing)
    simulation_notes TEXT,             -- Optional metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- WEC simulation data 
CREATE TABLE wec_simulations (
    wec_sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER,
    wec_farm_name TEXT NOT NULL,
    wecsim_case TEXT,
    power_output_mw REAL,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id)
);
```

### PSS®E-Specific Tables (following GridState schema)

```sql
-- PSS®E Bus Results
CREATE TABLE psse_bus_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    bus INTEGER NOT NULL,              -- Bus number
    bus_name TEXT,                     -- e.g., "Bus_1"
    type TEXT,                         -- "Slack", "PV", "PQ"
    p REAL,                           -- Net active power [pu on S_base]
    q REAL,                           -- Net reactive power [pu on S_base]
    v_mag REAL,                       -- Voltage magnitude [pu on V_base]
    angle_deg REAL,                   -- Voltage angle [degrees]
    vbase REAL,                       -- Bus nominal voltage [kV LL]
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, bus)
);

-- PSS®E Generator Results
CREATE TABLE psse_generator_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    gen INTEGER NOT NULL,              -- Generator ID
    gen_name TEXT,                     -- e.g., "Gen_1"
    bus INTEGER NOT NULL,              -- Connected bus
    p REAL,                           -- Active power output [pu on S_base]
    q REAL,                           -- Reactive power output [pu on S_base]
    mbase REAL,                       -- Generator nameplate [MVA]
    status INTEGER,                   -- 1=online, 0=offline
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, gen)
);

-- PSS®E Load Results
CREATE TABLE psse_load_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    load INTEGER NOT NULL,             -- Load ID
    load_name TEXT,                    -- e.g., "Load_1"
    bus INTEGER NOT NULL,              -- Connected bus
    p REAL,                           -- Active power demand [pu on S_base]
    q REAL,                           -- Reactive power demand [pu on S_base]
    status INTEGER,                   -- 1=connected, 0=offline
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, load)
);

-- PSS®E Line Results
CREATE TABLE psse_line_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    line INTEGER NOT NULL,             -- Line ID
    line_name TEXT,                    -- e.g., "Line_1_2"
    ibus INTEGER NOT NULL,             -- From bus
    jbus INTEGER NOT NULL,             -- To bus
    line_pct REAL,                    -- Percentage of thermal rating [%]
    status INTEGER,                   -- 1=online, 0=offline
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, line)
);
```

### PyPSA-Specific Tables

```sql
-- PyPSA Bus Results (may have different/additional columns)
CREATE TABLE pypsa_bus_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    bus INTEGER NOT NULL,
    bus_name TEXT,
    type TEXT,
    p REAL,                           -- [pu on S_base]
    q REAL,                           -- [pu on S_base]
    v_mag REAL,                       -- [pu on V_base]
    angle_deg REAL,                   -- [degrees]
    vbase REAL,                       -- [kV LL]
    -- PyPSA-specific columns
    marginal_price REAL,              -- Locational marginal price [$/MWh]
    carrier TEXT,                     -- Energy carrier type
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, bus)
);

-- PyPSA Generator Results
CREATE TABLE pypsa_generator_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    gen INTEGER NOT NULL,
    gen_name TEXT,
    bus INTEGER NOT NULL,
    p REAL,                           -- [pu on S_base]
    q REAL,                           -- [pu on S_base]
    mbase REAL,                       -- [MVA]
    status INTEGER,
    -- PyPSA-specific columns
    p_nom REAL,                       -- Nominal power [MW]
    p_min_pu REAL,                   -- Minimum output [pu of p_nom]
    p_max_pu REAL,                   -- Maximum output [pu of p_nom]
    efficiency REAL,                 -- Generator efficiency [pu]
    carrier TEXT,                    -- Technology type
    capital_cost REAL,               -- Capital cost [$/MW]
    marginal_cost REAL,              -- Marginal cost [$/MWh]
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, gen)
);

-- PyPSA Link Results (PyPSA-specific component)
CREATE TABLE pypsa_link_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    link INTEGER NOT NULL,             -- Link ID
    link_name TEXT,
    bus0 INTEGER NOT NULL,             -- From bus
    bus1 INTEGER NOT NULL,             -- To bus
    p0 REAL,                          -- Power at bus0 [pu on S_base]
    p1 REAL,                          -- Power at bus1 [pu on S_base]
    efficiency REAL,                  -- Link efficiency [pu]
    status INTEGER,
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id),
    UNIQUE(grid_sim_id, timestamp, link)
);
```

### Database Access Pattern

```python
# Example: Insert PSS®E simulation data
def insert_psse_data(grid_sim_id: int, timestamp: str, grid_state: GridState):
    """Insert GridState data into PSS®E-specific tables."""
    
    # Bus data
    for _, row in grid_state.bus.iterrows():
        cursor.execute("""
            INSERT INTO psse_bus_results 
            (grid_sim_id, timestamp, bus, bus_name, type, p, q, v_mag, angle_deg, vbase)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (grid_sim_id, timestamp, row.name, row['bus_name'], row['type'], 
              row['p'], row['q'], row['v_mag'], row['angle_deg'], row['Vbase']))
    
    # Generator data
    for _, row in grid_state.gen.iterrows():
        cursor.execute("""
            INSERT INTO psse_generator_results 
            (grid_sim_id, timestamp, gen, gen_name, bus, p, q, mbase, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (grid_sim_id, timestamp, row.name, row['gen_name'], row['bus'],
              row['p'], row['q'], row['Mbase'], row['status']))

# Example: Query data regardless of software
def get_all_bus_data(grid_sim_id: int) -> pd.DataFrame:
    """Get bus data for any software backend."""
    
    # Determine software type
    software = cursor.execute("""
        SELECT software FROM grid_simulations WHERE grid_sim_id = ?
    """, (grid_sim_id,)).fetchone()[0]
    
    # Query appropriate table
    if software == "PSSE":
        table = "psse_bus_results"
    elif software == "PyPSA":
        table = "pypsa_bus_results"
    else:
        raise ValueError(f"Unknown software: {software}")
    
    return pd.read_sql_query(f"""
        SELECT * FROM {table} WHERE grid_sim_id = ? ORDER BY timestamp, bus
    """, conn, params=(grid_sim_id,))
```

### Key Benefits of This Approach:

1. **Clear Separation**: Each software's data is in dedicated tables, avoiding confusion
2. **Flexible Schema**: Each software can have unique columns without affecting others
3. **Common Interface**: All tables share the same `grid_sim_id` for linking
4. **GridState Alignment**: Core columns match your GridState schema exactly
5. **Extensible**: Easy to add new software backends (GridLAB-D, PowerWorld, etc.)

### Naming Convention:
- `{software}_bus_results`
- `{software}_generator_results` 
- `{software}_load_results`
- `{software}_line_results`
- `{software}_{unique_component}_results` (for software-specific components)

This way, when you load PSS®E data, it goes to `psse_*` tables. When you load PyPSA data with the same `grid_sim_id`, it goes to `pypsa_*` tables. No confusion, and each software's unique capabilities are preserved!

Would you like me to extend this schema for any other specific software backends you're planning to support?
