# Database

The WEC-Grid database component provides persistent storage and retrieval of simulation data, model configurations, and results.

## Overview

The database system in WEC-Grid handles:

- SQLite-based data storage
- Simulation result persistence
- Model configuration storage
- Time-series data management
- Query and retrieval operations

## Features

- **Automatic Schema Management**: Database tables are created and managed automatically
- **Result Storage**: Power flow results, time-series data, and WEC outputs
- **Model Persistence**: Save and load grid models and WEC configurations
- **Query Interface**: Flexible data retrieval with filtering and aggregation
- **Backup & Export**: Database backup and data export capabilities

## Database Structure

The WEC-Grid database includes tables for:

- `simulations`: Simulation metadata and configurations
- `power_flow_results`: Power flow analysis results
- `wec_data`: WEC device and farm data
- `time_series`: Time-varying simulation data
- `grid_models`: Power system model information

## Basic Usage

```python
from wecgrid.database import WECGridDB

# Initialize database connection
db = WECGridDB()

# Store simulation results
db.store_simulation_results(simulation_id, results_data)

# Retrieve data
data = db.get_simulation_data(simulation_id)

# Query time series
ts_data = db.query_time_series(start_time, end_time)
```

## Integration with Engine

The database is automatically used by the Engine for result storage:

```python
import wecgrid

engine = wecgrid.Engine()
# Database operations happen automatically
results = engine.run_simulation()
# Results are automatically stored in database
```

## API Reference

![mkapi](wecgrid.database.wecgrid_db.WECGridDB)