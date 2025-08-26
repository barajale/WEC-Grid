# WECFarm & WECDevice
<!-- 
Wave Energy Converter modeling for individual devices and farm arrays.

## Features

- Individual WEC device modeling
- WEC farm arrays with multiple devices
- Built-in models (RM3, custom models)
- Grid connection and power injection

## Basic Usage

```python
# Add WEC farm to simulation
engine.apply_wec(
    farm_name="CoastalFarm",
    size=8,              # Number of devices
    model="RM3",         # Built-in or custom model
    bus_location=31,     # Grid connection point
    sim_id=1            # WECSim results ID
)
```

## API Reference

![mkapi](wecgrid.wec.wecfarm.WECFarm)

![mkapi](wecgrid.wec.wecdevice.WECDevice) -->