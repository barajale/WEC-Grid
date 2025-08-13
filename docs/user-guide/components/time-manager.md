# Time Manager

The Time Manager component coordinates temporal aspects across all WEC-Grid simulations, ensuring synchronized timing between PSS®E, PyPSA, and WEC-Sim platforms.

## Overview

The `WECGridTimeManager` provides centralized time coordination for multi-platform simulations, managing:

- Simulation start and stop times
- Time step intervals and frequency
- Snapshot generation for time series data
- Temporal synchronization across platforms

## Key Features

- **Unified timing**: Single source of truth for all simulation timing
- **Flexible intervals**: Support for various time step sizes (minutes to hours)
- **Pandas integration**: Compatible with pandas DatetimeIndex for time series operations
- **Automatic calculation**: Derives end times from duration and frequency parameters

## Usage

```python
from datetime import datetime
from wecgrid.util import WECGridTimeManager

# Create time manager for 24-hour simulation
time_mgr = WECGridTimeManager(
    start_time=datetime(2023, 1, 1, 0, 0, 0),
    sim_length=288,  # 24 hours at 5-minute intervals
    freq="5T"
)

# Access time snapshots
snapshots = time_mgr.snapshots
print(f"Simulation duration: {len(snapshots)} time steps")
```

## Configuration

Time parameters can be updated dynamically:

```python
# Change to hourly intervals
time_mgr.update(freq="1H", sim_length=24)

# Set specific end time
end_time = datetime(2023, 1, 2, 0, 0, 0)
time_mgr.set_end_time(end_time)
```

## API Reference

![mkapi](wecgrid.util.wecgrid_timemanager.WECGridTimeManager)
