-- ============================================================================
-- WEC-Grid Complete Database Schema
-- Multi-Software Backend Support with GridState Alignment
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Core Simulation Tables
-- ----------------------------------------------------------------------------

-- Main simulation metadata (shared across all software)
CREATE TABLE grid_simulations (
    grid_sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_name TEXT NOT NULL,                    -- e.g., "IEEE_14_bus"
    software TEXT NOT NULL,                     -- "PSSE", "PyPSA", "GridLAB-D", etc.
    sbase_mva REAL NOT NULL,                   -- System base MVA (e.g., 100)
    timestamp_start TEXT NOT NULL,             -- ISO format timestamp
    timestamp_end TEXT,                        -- ISO format timestamp (NULL if ongoing)
    simulation_notes TEXT,                     -- Optional metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique simulation per case/software combination
    UNIQUE(case_name, software, timestamp_start)
);

-- WEC simulation data (works with any grid software)
CREATE TABLE wec_simulations (
    wec_sim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    wec_farm_name TEXT NOT NULL,
    wecsim_case TEXT,
    power_output_mw REAL,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE
);

-- Index for efficient time-series queries
CREATE INDEX idx_wec_simulations_time ON wec_simulations(grid_sim_id, timestamp);

-- ----------------------------------------------------------------------------
-- PSS®E-Specific Tables (Matching GridState Schema Exactly)
-- ----------------------------------------------------------------------------

-- PSS®E Bus Results
CREATE TABLE psse_bus_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Bus Schema Columns
    bus INTEGER NOT NULL,                      -- Bus number (unique identifier)
    bus_name TEXT,                            -- Bus name/label (e.g., "Bus_1", "Bus_2")
    type TEXT,                                -- Bus type: "Slack", "PV", "PQ"
    p REAL,                                   -- Net active power injection [pu on S_base]
    q REAL,                                   -- Net reactive power injection [pu on S_base]
    v_mag REAL,                               -- Voltage magnitude [pu on V_base]
    angle_deg REAL,                           -- Voltage angle [degrees]
    vbase REAL,                               -- Bus nominal voltage [kV LL]
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, bus)
);

-- PSS®E Generator Results
CREATE TABLE psse_generator_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Generator Schema Columns
    gen INTEGER NOT NULL,                      -- Generator ID
    gen_name TEXT,                            -- Generator name (e.g., "Gen_1")
    bus INTEGER NOT NULL,                     -- Connected bus number
    p REAL,                                   -- Active power output [pu on S_base]
    q REAL,                                   -- Reactive power output [pu on S_base]
    mbase REAL,                               -- Generator nameplate MVA rating
    status INTEGER,                           -- Generator status (1=online, 0=offline)
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, gen)
);

-- PSS®E Load Results
CREATE TABLE psse_load_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Load Schema Columns
    load INTEGER NOT NULL,                     -- Load ID
    load_name TEXT,                           -- Load name (e.g., "Load_1")
    bus INTEGER NOT NULL,                     -- Connected bus number
    p REAL,                                   -- Active power demand [pu on S_base]
    q REAL,                                   -- Reactive power demand [pu on S_base]
    status INTEGER,                           -- Load status (1=connected, 0=offline)
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, load)
);

-- PSS®E Line Results
CREATE TABLE psse_line_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Line Schema Columns
    line INTEGER NOT NULL,                     -- Line ID
    line_name TEXT,                           -- Line name (e.g., "Line_1_2")
    ibus INTEGER NOT NULL,                    -- From bus number
    jbus INTEGER NOT NULL,                    -- To bus number
    line_pct REAL,                            -- Percentage of thermal rating [%]
    status INTEGER,                           -- Line status (1=online, 0=offline)
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, line)
);

-- ----------------------------------------------------------------------------
-- PyPSA-Specific Tables (Core GridState + PyPSA Extensions)
-- ----------------------------------------------------------------------------

-- PyPSA Bus Results
CREATE TABLE pypsa_bus_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Bus Schema Columns (Core)
    bus INTEGER NOT NULL,
    bus_name TEXT,
    type TEXT,
    p REAL,                                   -- [pu on S_base]
    q REAL,                                   -- [pu on S_base]
    v_mag REAL,                               -- [pu on V_base]
    angle_deg REAL,                           -- [degrees]
    vbase REAL,                               -- [kV LL]
    
    -- PyPSA-Specific Extensions
    marginal_price REAL,                      -- Locational marginal price [$/MWh]
    carrier TEXT,                             -- Energy carrier type
    x REAL,                                   -- Bus x-coordinate
    y REAL,                                   -- Bus y-coordinate
    country TEXT,                             -- Country code
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, bus)
);

-- PyPSA Generator Results
CREATE TABLE pypsa_generator_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Generator Schema Columns (Core)
    gen INTEGER NOT NULL,
    gen_name TEXT,
    bus INTEGER NOT NULL,
    p REAL,                                   -- [pu on S_base]
    q REAL,                                   -- [pu on S_base]
    mbase REAL,                               -- [MVA]
    status INTEGER,
    
    -- PyPSA-Specific Extensions
    p_nom REAL,                               -- Nominal power [MW]
    p_min_pu REAL,                           -- Minimum output [pu of p_nom]
    p_max_pu REAL,                           -- Maximum output [pu of p_nom]
    efficiency REAL,                         -- Generator efficiency [pu]
    carrier TEXT,                            -- Technology type
    capital_cost REAL,                       -- Capital cost [$/MW]
    marginal_cost REAL,                      -- Marginal cost [$/MWh]
    ramp_limit_up REAL,                      -- Ramp limit up [MW/h]
    ramp_limit_down REAL,                    -- Ramp limit down [MW/h]
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, gen)
);

-- PyPSA Load Results
CREATE TABLE pypsa_load_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Load Schema Columns (Core)
    load INTEGER NOT NULL,
    load_name TEXT,
    bus INTEGER NOT NULL,
    p REAL,                                   -- [pu on S_base]
    q REAL,                                   -- [pu on S_base]
    status INTEGER,
    
    -- PyPSA-Specific Extensions
    carrier TEXT,                             -- Load type/carrier
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, load)
);

-- PyPSA Line Results
CREATE TABLE pypsa_line_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Line Schema Columns (Core)
    line INTEGER NOT NULL,
    line_name TEXT,
    ibus INTEGER NOT NULL,                    -- From bus (bus0 in PyPSA)
    jbus INTEGER NOT NULL,                    -- To bus (bus1 in PyPSA)
    line_pct REAL,                            -- Percentage of thermal rating [%]
    status INTEGER,
    
    -- PyPSA-Specific Extensions
    p0 REAL,                                  -- Power flow at bus0 [pu on S_base]
    p1 REAL,                                  -- Power flow at bus1 [pu on S_base]
    q0 REAL,                                  -- Reactive power at bus0 [pu on S_base]
    q1 REAL,                                  -- Reactive power at bus1 [pu on S_base]
    s_nom REAL,                               -- Nominal apparent power [MVA]
    length REAL,                              -- Line length [km]
    r REAL,                                   -- Resistance [pu]
    x REAL,                                   -- Reactance [pu]
    b REAL,                                   -- Susceptance [pu]
    capital_cost REAL,                        -- Capital cost [$/MW/km]
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, line)
);

-- PyPSA Link Results (PyPSA-specific component, no GridState equivalent)
CREATE TABLE pypsa_link_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    link INTEGER NOT NULL,                     -- Link ID
    link_name TEXT,                           -- Link name
    bus0 INTEGER NOT NULL,                    -- From bus
    bus1 INTEGER NOT NULL,                    -- To bus
    p0 REAL,                                  -- Power at bus0 [pu on S_base]
    p1 REAL,                                  -- Power at bus1 [pu on S_base]
    efficiency REAL,                          -- Link efficiency [pu]
    p_nom REAL,                               -- Nominal power [MW]
    status INTEGER,                           -- Link status (1=online, 0=offline)
    carrier TEXT,                             -- Technology type
    capital_cost REAL,                        -- Capital cost [$/MW]
    marginal_cost REAL,                       -- Marginal cost [$/MWh]
    length REAL,                              -- Link length [km]
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, link)
);

-- PyPSA Storage Results (PyPSA-specific component)
CREATE TABLE pypsa_storage_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    storage INTEGER NOT NULL,                  -- Storage ID
    storage_name TEXT,                        -- Storage name
    bus INTEGER NOT NULL,                     -- Connected bus
    p REAL,                                   -- Power output [pu on S_base]
    state_of_charge REAL,                     -- State of charge [MWh]
    p_nom REAL,                               -- Nominal power [MW]
    max_hours REAL,                           -- Maximum storage duration [hours]
    efficiency_store REAL,                    -- Storage efficiency [pu]
    efficiency_dispatch REAL,                 -- Dispatch efficiency [pu]
    status INTEGER,                           -- Storage status (1=online, 0=offline)
    carrier TEXT,                             -- Storage technology
    capital_cost REAL,                        -- Capital cost [$/MW]
    marginal_cost REAL,                       -- Marginal cost [$/MWh]
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, storage)
);

-- ----------------------------------------------------------------------------
-- GridLAB-D-Specific Tables (Future Extension)
-- ----------------------------------------------------------------------------

-- GridLAB-D Bus Results
CREATE TABLE gridlabd_bus_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_sim_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    
    -- GridState Bus Schema Columns (Core)
    bus INTEGER NOT NULL,
    bus_name TEXT,
    type TEXT,
    p REAL,                                   -- [pu on S_base]
    q REAL,                                   -- [pu on S_base]
    v_mag REAL,                               -- [pu on V_base]
    angle_deg REAL,                           -- [degrees]
    vbase REAL,                               -- [kV LL]
    
    -- GridLAB-D-Specific Extensions
    voltage_a_real REAL,                      -- Phase A voltage real [V]
    voltage_a_imag REAL,                      -- Phase A voltage imaginary [V]
    voltage_b_real REAL,                      -- Phase B voltage real [V]
    voltage_b_imag REAL,                      -- Phase B voltage imaginary [V]
    voltage_c_real REAL,                      -- Phase C voltage real [V]
    voltage_c_imag REAL,                      -- Phase C voltage imaginary [V]
    frequency REAL,                           -- System frequency [Hz]
    
    FOREIGN KEY (grid_sim_id) REFERENCES grid_simulations(grid_sim_id) ON DELETE CASCADE,
    UNIQUE(grid_sim_id, timestamp, bus)
);

-- ----------------------------------------------------------------------------
-- Performance Indexes for Time-Series Queries
-- ----------------------------------------------------------------------------

-- PSS®E Indexes
CREATE INDEX idx_psse_bus_time ON psse_bus_results(grid_sim_id, timestamp);
CREATE INDEX idx_psse_bus_component ON psse_bus_results(bus);
CREATE INDEX idx_psse_gen_time ON psse_generator_results(grid_sim_id, timestamp);
CREATE INDEX idx_psse_gen_component ON psse_generator_results(gen);
CREATE INDEX idx_psse_load_time ON psse_load_results(grid_sim_id, timestamp);
CREATE INDEX idx_psse_load_component ON psse_load_results(load);
CREATE INDEX idx_psse_line_time ON psse_line_results(grid_sim_id, timestamp);
CREATE INDEX idx_psse_line_component ON psse_line_results(line);

-- PyPSA Indexes
CREATE INDEX idx_pypsa_bus_time ON pypsa_bus_results(grid_sim_id, timestamp);
CREATE INDEX idx_pypsa_bus_component ON pypsa_bus_results(bus);
CREATE INDEX idx_pypsa_gen_time ON pypsa_generator_results(grid_sim_id, timestamp);
CREATE INDEX idx_pypsa_gen_component ON pypsa_generator_results(gen);
CREATE INDEX idx_pypsa_load_time ON pypsa_load_results(grid_sim_id, timestamp);
CREATE INDEX idx_pypsa_load_component ON pypsa_load_results(load);
CREATE INDEX idx_pypsa_line_time ON pypsa_line_results(grid_sim_id, timestamp);
CREATE INDEX idx_pypsa_line_component ON pypsa_line_results(line);
CREATE INDEX idx_pypsa_link_time ON pypsa_link_results(grid_sim_id, timestamp);
CREATE INDEX idx_pypsa_link_component ON pypsa_link_results(link);
CREATE INDEX idx_pypsa_storage_time ON pypsa_storage_results(grid_sim_id, timestamp);
CREATE INDEX idx_pypsa_storage_component ON pypsa_storage_results(storage);

-- GridLAB-D Indexes
CREATE INDEX idx_gridlabd_bus_time ON gridlabd_bus_results(grid_sim_id, timestamp);
CREATE INDEX idx_gridlabd_bus_component ON gridlabd_bus_results(bus);

-- ----------------------------------------------------------------------------
-- Views for Cross-Software Compatibility
-- ----------------------------------------------------------------------------

-- Universal Bus View (Common columns across all software)
CREATE VIEW universal_bus_results AS
SELECT 
    'PSSE' as software,
    grid_sim_id,
    timestamp,
    bus,
    bus_name,
    type,
    p,
    q,
    v_mag,
    angle_deg,
    vbase
FROM psse_bus_results

UNION ALL

SELECT 
    'PyPSA' as software,
    grid_sim_id,
    timestamp,
    bus,
    bus_name,
    type,
    p,
    q,
    v_mag,
    angle_deg,
    vbase
FROM pypsa_bus_results

UNION ALL

SELECT 
    'GridLAB-D' as software,
    grid_sim_id,
    timestamp,
    bus,
    bus_name,
    type,
    p,
    q,
    v_mag,
    angle_deg,
    vbase
FROM gridlabd_bus_results;

-- Universal Generator View
CREATE VIEW universal_generator_results AS
SELECT 
    'PSSE' as software,
    grid_sim_id,
    timestamp,
    gen,
    gen_name,
    bus,
    p,
    q,
    mbase,
    status
FROM psse_generator_results

UNION ALL

SELECT 
    'PyPSA' as software,
    grid_sim_id,
    timestamp,
    gen,
    gen_name,
    bus,
    p,
    q,
    mbase,
    status
FROM pypsa_generator_results;

-- Universal Load View
CREATE VIEW universal_load_results AS
SELECT 
    'PSSE' as software,
    grid_sim_id,
    timestamp,
    load,
    load_name,
    bus,
    p,
    q,
    status
FROM psse_load_results

UNION ALL

SELECT 
    'PyPSA' as software,
    grid_sim_id,
    timestamp,
    load,
    load_name,
    bus,
    p,
    q,
    status
FROM pypsa_load_results;

-- Universal Line View
CREATE VIEW universal_line_results AS
SELECT 
    'PSSE' as software,
    grid_sim_id,
    timestamp,
    line,
    line_name,
    ibus,
    jbus,
    line_pct,
    status
FROM psse_line_results

UNION ALL

SELECT 
    'PyPSA' as software,
    grid_sim_id,
    timestamp,
    line,
    line_name,
    ibus,
    jbus,
    line_pct,
    status
FROM pypsa_line_results;

-- ----------------------------------------------------------------------------
-- Triggers for Data Validation
-- ----------------------------------------------------------------------------

-- Validate per-unit values are reasonable
CREATE TRIGGER validate_bus_pu_values 
BEFORE INSERT ON psse_bus_results
FOR EACH ROW
WHEN NEW.p < -10 OR NEW.p > 10 OR NEW.q < -10 OR NEW.q > 10 OR NEW.v_mag < 0.5 OR NEW.v_mag > 1.5
BEGIN
    SELECT RAISE(ABORT, 'Bus per-unit values out of reasonable range');
END;

-- Validate generator status
CREATE TRIGGER validate_generator_status
BEFORE INSERT ON psse_generator_results
FOR EACH ROW
WHEN NEW.status NOT IN (0, 1)
BEGIN
    SELECT RAISE(ABORT, 'Generator status must be 0 or 1');
END;

-- Apply same validation to PyPSA tables
CREATE TRIGGER validate_pypsa_bus_pu_values 
BEFORE INSERT ON pypsa_bus_results
FOR EACH ROW
WHEN NEW.p < -10 OR NEW.p > 10 OR NEW.q < -10 OR NEW.q > 10 OR NEW.v_mag < 0.5 OR NEW.v_mag > 1.5
BEGIN
    SELECT RAISE(ABORT, 'PyPSA Bus per-unit values out of reasonable range');
END;

CREATE TRIGGER validate_pypsa_generator_status
BEFORE INSERT ON pypsa_generator_results
FOR EACH ROW
WHEN NEW.status NOT IN (0, 1)
BEGIN
    SELECT RAISE(ABORT, 'PyPSA Generator status must be 0 or 1');
END;

-- ----------------------------------------------------------------------------
-- Example Usage Queries
-- ----------------------------------------------------------------------------

/*
-- Insert a new grid simulation
INSERT INTO grid_simulations (case_name, software, sbase_mva, timestamp_start)
VALUES ('IEEE_14_bus', 'PSSE', 100.0, '2025-08-14T10:00:00');

-- Insert PSS®E bus results from GridState
INSERT INTO psse_bus_results (grid_sim_id, timestamp, bus, bus_name, type, p, q, v_mag, angle_deg, vbase)
VALUES (1, '2025-08-14T10:01:00', 1, 'Bus_1', 'Slack', 2.32, 0.16, 1.06, 0.0, 138.0);

-- Query voltage magnitude time series for all buses
SELECT timestamp, bus, v_mag 
FROM universal_bus_results 
WHERE grid_sim_id = 1 
ORDER BY timestamp, bus;

-- Compare voltage magnitudes between PSS®E and PyPSA for same case
SELECT 
    p.timestamp,
    p.bus,
    p.v_mag as psse_voltage,
    y.v_mag as pypsa_voltage,
    ABS(p.v_mag - y.v_mag) as voltage_diff
FROM psse_bus_results p
JOIN pypsa_bus_results y ON p.bus = y.bus AND p.timestamp = y.timestamp
WHERE p.grid_sim_id = 1 AND y.grid_sim_id = 2
ORDER BY voltage_diff DESC;

-- Get WEC farm power output correlated with grid state
SELECT 
    w.timestamp,
    w.wec_farm_name,
    w.power_output_mw,
    b.v_mag as bus_voltage,
    g.p as gen_output
FROM wec_simulations w
JOIN psse_bus_results b ON w.grid_sim_id = b.grid_sim_id AND w.timestamp = b.timestamp
JOIN psse_generator_results g ON w.grid_sim_id = g.grid_sim_id AND w.timestamp = g.timestamp
WHERE w.grid_sim_id = 1 AND b.bus = 14 AND g.gen = 1
ORDER BY w.timestamp;
*/
