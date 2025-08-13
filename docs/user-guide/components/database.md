# Database

SQLite-based storage for simulation results, configurations, and time-series data.

## Features

- Automatic result storage during simulations
- Query interface for data retrieval  
- Export capabilities for external analysis

## Basic Usage

```python
# Results are stored automatically
engine = wecgrid.Engine()
engine.simulate()  # Results saved to database

# Query stored data
db = engine.database
data = db.get_simulation_data(simulation_id)
```

## API Reference

![mkapi](wecgrid.database.wecgrid_db.WECGridDB)