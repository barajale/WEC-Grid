# Time Manager

Coordinates simulation timing across all platforms (PSS®E, PyPSA, WEC-Sim) for synchronized execution.

## Features

- Centralized time coordination
- Flexible time step intervals (5-minute to hourly)
- Automatic snapshot generation
- Pandas DatetimeIndex integration

## Basic Usage

```python
# Time manager is handled automatically by Engine
engine = wecgrid.Engine()
engine.simulate()  # Uses default 24-hour simulation with 5-minute steps

# Access time information
snapshots = engine.time.snapshots
print(f"Simulation duration: {len(snapshots)} time steps")
```

## API Reference

![mkapi](wecgrid.util.time.WECGridTime)
