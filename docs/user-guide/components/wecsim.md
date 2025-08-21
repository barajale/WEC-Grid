# WECSim Integration

High-fidelity wave energy converter modeling through MATLAB WECSim integration.

## Requirements

- MATLAB with Simulink
- WECSim toolbox installed
- MATLAB Engine API for Python

## Basic Usage

```python
# Configure WECSim path
engine.wec_sim.wec_sim_path = r"C:\Users\me\WEC-Sim"

# Run WECSim simulation
engine.wec_sim(
    sim_id=1,
    model="RM3",
    sim_length_secs=3600,
    wave_height=2.5,
    wave_period=8.0
)

# Apply results to grid simulation
engine.apply_wec(sim_id=1, model="RM3", bus_location=31)
```

## API Reference

![mkapi](wecgrid.wec.wecsim_runner.WECSimRunner)