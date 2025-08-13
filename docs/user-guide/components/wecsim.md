# WECSim Integration

WECSim (Wave Energy Converter Simulator) integration provides high-fidelity wave energy converter modeling within WEC-Grid simulations.

## Overview

The WECSim integration enables:

- **High-Fidelity WEC Modeling**: Detailed hydrodynamic and power-take-off modeling
- **MATLAB/Simulink Integration**: Leverage WECSim's native MATLAB environment
- **Automated Workflow**: Seamless data exchange between WECSim and WEC-Grid
- **Batch Processing**: Run multiple WECSim cases for different sea states
- **Result Integration**: Incorporate WECSim outputs into power system analysis

## Requirements

- MATLAB with Simulink installed
- WECSim toolbox properly configured
- Python-MATLAB engine (optional for direct integration)

## Workflow

1. **WEC Model Setup**: Configure WECSim models with appropriate parameters
2. **Sea State Definition**: Define wave conditions for simulation
3. **Simulation Execution**: Run WECSim through WEC-Grid interface
4. **Result Processing**: Extract power output and convert to grid-compatible format
5. **Grid Integration**: Use WEC outputs in power system analysis

## Basic Usage

```python
from wecgrid.wec import WECSimRunner

# Initialize WECSim runner
wecsim = WECSimRunner()

# Configure WEC model
wecsim.set_wec_model("RM3")
wecsim.set_sea_state(Hs=2.5, Tp=8.0)

# Run simulation
results = wecsim.run_simulation(duration=3600)

# Extract power output
power_output = wecsim.get_power_time_series()
```

## Integration with WEC-Grid

WECSim is integrated through the WEC farm modeling:

```python
import wecgrid

engine = wecgrid.Engine()

# Set up WEC farm with WECSim models
wec_farm = engine.create_wec_farm()
wec_farm.add_device("RM3", location=(0, 0))
wec_farm.set_wecsim_model("RM3_model.slx")

# Run coupled simulation
results = engine.run_simulation()
```

## Model Configuration

WECSim models can be configured through WEC-Grid:

```python
# Set WEC parameters
wecsim.configure_wec({
    'body_mass': 727010,
    'body_volume': 541,
    'pto_damping': 1200000,
    'pto_stiffness': 0
})

# Define environmental conditions
wecsim.set_environment({
    'water_depth': 50,
    'wave_spectrum': 'PM',
    'significant_height': 2.5,
    'peak_period': 8.0
})
```

## API Reference

![mkapi](wecgrid.wec.wecsim_runner.WECSimRunner)