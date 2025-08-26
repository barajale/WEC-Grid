# Engine

Central coordinator for WEC-Grid simulations, managing power system backends, WEC farms, and data flow.
<!-- 
## Quick Start

```python
import wecgrid

# Initialize engine and load grid case
engine = wecgrid.Engine()
engine.case("IEEE_30_bus")

# Load power system backends
engine.load(["psse", "pypsa"])

# Add WEC farm
engine.apply_wec(
    farm_name="CoastalFarm",
    size=8,
    sim_id=1,
    model="RM3",
    bus_location=31
)

# Run simulation
engine.simulate()
```

## Key Methods

- **`engine.case()`** - Load grid model (IEEE test systems or custom .RAW files)
- **`engine.load()`** - Initialize power system backends (PSS®E, PyPSA)
- **`engine.apply_wec()`** - Add WEC farms to the simulation
- **`engine.simulate()`** - Run coordinated simulation across all backends

## Configuration Options

```python
# Custom simulation parameters
engine.simulate(
    load_curve=True,  # Apply synthetic load profiles
    plot=True        # Generate result plots
)

# Generate load curves
engine.generate_load_curves(
    morning_peak_hour=8.0,
    evening_peak_hour=18.0,
    amplitude=0.30
)
```

## API Reference

![mkapi](wecgrid.engine.Engine) -->
