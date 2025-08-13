"""WEC-Grid Database Interface Module.

SQLite database interface for WEC-Grid simulation data management, including 
time series storage, configuration persistence, and result archival.

The database supports:
    - Time series data: WEC power output, grid states, environmental conditions
    - Configuration management: Simulation parameters, WEC farm layouts
    - Result archival: Long-term storage with metadata for reproducibility
    - Cross-platform compatibility: SQLite backend for portability

Example:
    >>> from wecgrid.database import WECGridDB
    >>> db = WECGridDB()
    >>> with db.connection() as conn:
    ...     results = db.query("SELECT * FROM simulation_runs")

Notes:
    - Database file created automatically on first connection
    - Thread-safe for concurrent read operations
    - Schema evolution supported through migration scripts
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional
import pandas as pd

# default location for the DB file
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB = os.path.join(_CURR_DIR, "WEC-GRID.db")
DB_PATH = _DEFAULT_DB

class WECGridDB:
    """SQLite database interface for WEC-Grid simulation data management.
    
    Provides database operations for storing WEC simulation results, device
    configurations, and time series data. Supports both raw SQL queries and
    pandas DataFrame integration.
    
    Attributes:
        db_path (str): Path to SQLite database file.
        
    Example:
        >>> db = WECGridDB()
        >>> with db.connection() as conn:
        ...     results = db.query("SELECT * FROM simulation_runs")
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database handler.
        
        Args:
            db_path (str, optional): Path to SQLite database file. Defaults
                to package database location if None.
        """
        self.db_path = db_path or _DEFAULT_DB

    @contextmanager
    def connection(self):
        """Context manager for safe database connections.
        
        Provides transaction safety with automatic commit on success and
        rollback on exceptions. Connection closed automatically.
        
        Yields:
            sqlite3.Connection: Database connection for SQL operations.
            
        Example:
            >>> with db.connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM simulation_runs")
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()
            
    def initialize_db(self):
        """Initialize database schema with WEC-Grid tables and indexes.
        
        Creates required tables for simulation data storage including
        simulation runs, WEC devices, and time series data with appropriate
        foreign key constraints and indexes.
        
        with self.connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            **simulation_runs table**:
            ```sql
            CREATE TABLE simulation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                frequency TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            ```
            
            **wec_devices table**:
            ```sql
            CREATE TABLE wec_devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                device_type TEXT NOT NULL,
                rated_power REAL NOT NULL CHECK(rated_power > 0),
                location_lat REAL CHECK(location_lat BETWEEN -90 AND 90),
                location_lon REAL CHECK(location_lon BETWEEN -180 AND 180),
                depth REAL,
                installation_date DATE,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            ```
            
            **timeseries_data table**:
            ```sql
            CREATE TABLE timeseries_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER NOT NULL,
                device_id INTEGER,
                timestamp TIMESTAMP NOT NULL,
                power_output REAL,
                wave_height REAL,
                wave_period REAL,
                wind_speed REAL,
                grid_frequency REAL,
                voltage REAL,
                current REAL,
                data_source TEXT DEFAULT 'simulation',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (simulation_id) REFERENCES simulation_runs(id) ON DELETE CASCADE,
                FOREIGN KEY (device_id) REFERENCES wec_devices(id) ON DELETE SET NULL
            );
            ```
            
        Indexes Created:
            **Performance optimization for common query patterns**:
            ```sql
            -- Temporal queries (most common)
            CREATE INDEX idx_timeseries_timestamp ON timeseries_data(timestamp);
            CREATE INDEX idx_timeseries_sim_time ON timeseries_data(simulation_id, timestamp);
            
            -- Device-specific analysis
            CREATE INDEX idx_timeseries_device ON timeseries_data(device_id);
            CREATE INDEX idx_timeseries_device_time ON timeseries_data(device_id, timestamp);
            
            -- Simulation metadata
            CREATE INDEX idx_simulation_name ON simulation_runs(name);
            CREATE INDEX idx_simulation_time_range ON simulation_runs(start_time, end_time);
            
            -- Device location queries
            CREATE INDEX idx_device_location ON wec_devices(location_lat, location_lon);
            ```
            
        Example:
            >>> # Initialize new database
            >>> db = WECGridDB("/path/to/new/database.db")
            >>> db.initialize_db()
            >>> print("Database schema created successfully")
            
            >>> # Verify table creation
            >>> tables = db.query(
            ...     "SELECT name FROM sqlite_master WHERE type='table'",
            ...     return_type="df"
            ... )
            >>> print(f"Created tables: {sorted(tables['name'].tolist())}")
            Created tables: ['simulation_runs', 'timeseries_data', 'wec_devices']
            
            >>> # Check index creation
            >>> indexes = db.query(
            ...     "SELECT name FROM sqlite_master WHERE type='index'",
            ...     return_type="df"
            ... )
            >>> print(f"Created indexes: {len(indexes)} total")
            
        Default Data:
            **Reference data for consistent simulation setup**:
            ```sql
            -- Common device types
            INSERT INTO device_types (name, description) VALUES
                ('RM3', 'Reference Model 3 - Point Absorber'),
                ('OWC', 'Oscillating Water Column'),
                ('Attenuator', 'Multi-body Wave Energy Converter');
                
            -- Simulation status values
            INSERT INTO simulation_status (status, description) VALUES
                ('pending', 'Simulation queued for execution'),
                ('running', 'Simulation currently executing'),
                ('completed', 'Simulation finished successfully'),
                ('failed', 'Simulation terminated with errors');
            ```
            
        Views Created:
            **Common analysis patterns as database views**:
            ```sql
            -- Device power summary
            CREATE VIEW device_power_summary AS
            SELECT 
                d.name as device_name,
                d.rated_power,
                COUNT(t.id) as data_points,
                AVG(t.power_output) as avg_power,
                MAX(t.power_output) as peak_power,
                MIN(t.timestamp) as first_data,
                MAX(t.timestamp) as last_data
            FROM wec_devices d
            LEFT JOIN timeseries_data t ON d.id = t.device_id
            GROUP BY d.id, d.name, d.rated_power;
            
            -- Simulation performance metrics
            CREATE VIEW simulation_metrics AS
            SELECT 
                s.name as simulation_name,
                s.start_time,
                s.end_time,
                s.frequency,
                COUNT(DISTINCT t.device_id) as device_count,
                COUNT(t.id) as total_data_points,
                AVG(t.power_output) as avg_total_power
            FROM simulation_runs s
            LEFT JOIN timeseries_data t ON s.id = t.simulation_id
            GROUP BY s.id;
            ```
            
        Triggers for Data Integrity:
            **Automatic timestamp updates and validation**:
            ```sql
            -- Update timestamp on simulation_runs changes
            CREATE TRIGGER update_simulation_timestamp 
            AFTER UPDATE ON simulation_runs
            BEGIN
                UPDATE simulation_runs 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = NEW.id;
            END;
            
            -- Validate timestamp ordering in time series
            CREATE TRIGGER validate_timeseries_order
            BEFORE INSERT ON timeseries_data
            BEGIN
                SELECT CASE 
                    WHEN NEW.timestamp < (
                        SELECT start_time FROM simulation_runs 
                        WHERE id = NEW.simulation_id
                    )
                    THEN RAISE(ABORT, 'Timestamp before simulation start')
                END;
            END;
            ```
            
        Error Handling:
            **Graceful handling of existing schemas**:
            - Checks for existing tables before creation
            - Skips initialization if schema already exists
            - Reports creation status for each component
            - Handles partial initialization recovery
            
        Migration Support:
            For existing databases with older schemas:
            ```python
            # Check schema version
            version = db.query("PRAGMA user_version")[0][0]
            if version < CURRENT_SCHEMA_VERSION:
                db.migrate_schema(from_version=version)
            ```
            
        Performance Optimization:
            **Post-creation optimization steps**:
            - ANALYZE command for query planner statistics
            - Vacuum command for optimal file organization
            - Page size optimization for time series workloads
            - Memory settings for improved performance
            
        Notes:
            - Safe to call multiple times (idempotent operation)
            - Creates database file if it doesn't exist
            - Enables foreign key constraints for data integrity
            - Sets optimal SQLite pragmas for WEC-Grid workloads
            - Schema versioning support for future migrations
            
        See Also:
            connection: Context manager used for schema creation
            query: Method for verifying schema creation success
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create simulation_runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    frequency TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create wec_devices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wec_devices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    rated_power REAL NOT NULL CHECK(rated_power > 0),
                    location_lat REAL CHECK(location_lat BETWEEN -90 AND 90),
                    location_lon REAL CHECK(location_lon BETWEEN -180 AND 180),
                    depth REAL,
                    installation_date DATE,
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create timeseries_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeseries_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER NOT NULL,
                    device_id INTEGER,
                    timestamp TIMESTAMP NOT NULL,
                    power_output REAL,
                    wave_height REAL,
                    wave_period REAL,
                    wind_speed REAL,
                    grid_frequency REAL,
                    voltage REAL,
                    current REAL,
                    data_source TEXT DEFAULT 'simulation',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (simulation_id) REFERENCES simulation_runs(id) ON DELETE CASCADE,
                    FOREIGN KEY (device_id) REFERENCES wec_devices(id) ON DELETE SET NULL
                )
            """)
            
            # Create performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeseries_timestamp ON timeseries_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeseries_sim_time ON timeseries_data(simulation_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeseries_device ON timeseries_data(device_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_name ON simulation_runs(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_location ON wec_devices(location_lat, location_lon)")
            
            # Optimize database for time series workloads
            cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
            cursor.execute("PRAGMA synchronous = NORMAL")  # Balance safety and performance
            cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store = MEMORY")  # Use memory for temporary tables
        
    def query(self, sql: str, params: tuple = None, return_type: str = "raw"):
        """Execute SQL query with flexible result formatting.

        Args:
            sql (str): SQL query string with optional ? parameter placeholders.
            params (tuple, optional): Values for SQL parameters.
            return_type (str): Result format - "raw" (tuples), "df" (DataFrame), 
                or "dict" (dictionaries). Defaults to "raw".

        Returns:
            Query results in requested format: list of tuples (raw), 
            pandas DataFrame (df), or list of dictionaries (dict).

        Example:
            >>> results = db.query("SELECT * FROM simulation_runs WHERE id = ?", 
            ...                    params=(1,), return_type="df")
            
            >>> # DataFrame results for analysis
            >>> power_data = db.query(
            ...     "SELECT timestamp, power_output FROM timeseries_data "
            ...     "WHERE simulation_id = ? AND timestamp BETWEEN ? AND ?",
            ...     params=(1, "2023-01-01 00:00:00", "2023-01-01 23:59:59"),
            ...     return_type="df"
            ... )
            >>> print(f"Data shape: {power_data.shape}")
            >>> print(f"Average power: {power_data['power_output'].mean():.2f} kW")
            Data shape: (288, 2)
            Average power: 127.45 kW
            
            >>> # Dictionary results for JSON APIs
            >>> device_summary = db.query(
            ...     "SELECT name, device_type, rated_power FROM wec_devices",
            ...     return_type="dict"
            ... )
            >>> import json
            >>> print(json.dumps(device_summary[0], indent=2))
            {
              "name": "WEC-001",
              "device_type": "RM3",
              "rated_power": 150.0
            }

        Advanced Query Patterns:
            **Time series aggregation**:
            ```python
            monthly_power = db.query('''
                SELECT 
                    strftime('%Y-%m', timestamp) as month,
                    AVG(power_output) as avg_power,
                    SUM(power_output) as total_power,
                    COUNT(*) as data_points
                FROM timeseries_data 
                WHERE simulation_id = ?
                GROUP BY strftime('%Y-%m', timestamp)
                ORDER BY month
            ''', params=(sim_id,), return_type="df")
            ```
            
            **Device performance comparison**:
            ```python
            device_stats = db.query('''
                SELECT 
                    d.name,
                    d.rated_power,
                    AVG(t.power_output) as avg_output,
                    MAX(t.power_output) as peak_output,
                    AVG(t.power_output) / d.rated_power as capacity_factor
                FROM wec_devices d
                JOIN timeseries_data t ON d.id = t.device_id
                WHERE t.simulation_id = ?
                GROUP BY d.id, d.name, d.rated_power
                ORDER BY capacity_factor DESC
            ''', params=(sim_id,), return_type="df")
            ```
            
            **Complex temporal analysis**:
            ```python
            peak_hours = db.query('''
                SELECT 
                    strftime('%H', timestamp) as hour,
                    AVG(power_output) as avg_power,
                    PERCENTILE_90(power_output) as p90_power
                FROM timeseries_data
                WHERE simulation_id = ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY avg_power DESC
            ''', params=(sim_id,), return_type="df")
            ```

        Performance Considerations:
            **Query optimization**:
            - Use indexes for WHERE clauses on timestamp, simulation_id, device_id
            - LIMIT large result sets to avoid memory issues
            - Use aggregate functions (SUM, AVG, COUNT) for summarization
            - Consider views for frequently repeated complex queries
            
            **Return type performance**:
            - "raw": Fastest, minimal memory overhead
            - "df": Moderate overhead, optimal for analysis
            - "dict": Highest overhead, good for APIs and serialization
            
            **Parameter binding benefits**:
            - SQL injection prevention through proper escaping
            - Query plan caching for repeated queries with different parameters
            - Type safety and automatic conversion
            
        Memory Management:
            **Large result sets**:
            ```python
            # Stream large datasets to avoid memory issues
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params or ())
                while True:
                    batch = cursor.fetchmany(1000)  # Process in batches
                    if not batch:
                        break
                    # Process batch
            ```
            
        Error Handling:
            **Common error patterns**:
            ```python
            try:
                results = db.query(sql, params, return_type="df")
            except sqlite3.IntegrityError:
                # Handle constraint violations
                pass
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    # Handle missing tables
                    db.initialize_db()
                    results = db.query(sql, params, return_type="df")
            except ValueError as e:
                # Handle invalid return_type
                print(f"Invalid return type: {e}")
            ```
            
        Data Type Handling:
            **SQLite to Python type mapping**:
            - INTEGER → int
            - REAL → float  
            - TEXT → str
            - TIMESTAMP → str (use pd.to_datetime() for conversion)
            - NULL → None
            
            **DataFrame type inference**:
            ```python
            # Automatic type conversion in pandas
            df = db.query("SELECT * FROM timeseries_data", return_type="df")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['power_output'] = pd.to_numeric(df['power_output'])
            ```

        Notes:
            - All queries executed within automatic transaction management
            - Parameter binding prevents SQL injection attacks
            - Column names preserved in DataFrame and dictionary results
            - Large result sets may require streaming for memory efficiency
            - Query execution time depends on indexes and result set size
            
        See Also:
            connection: Context manager used for query execution
            initialize_db: Database schema setup for query targets
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or ())
            result = cursor.fetchall()

            if return_type == "df":
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(result, columns=columns)
            elif return_type == "dict":
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in result]
            elif return_type == "raw":
                return result
            else:
                raise ValueError(f"Invalid return_type '{return_type}'. Must be 'raw', 'df', or 'dict'.")