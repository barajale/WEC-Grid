# WEC Devices & Farms

The WEC (Wave Energy Converter) module provides modeling capabilities for individual WEC devices and WEC farms within the power system analysis.

## Overview

The WEC modeling system includes:

- **Individual WEC Devices**: Single wave energy converter modeling
- **WEC Farms**: Multiple WEC devices with array interactions
- **Power Output Models**: Simplified and detailed power generation models
- **Grid Integration**: Connection to power system buses
- **Economic Modeling**: Cost and revenue calculations

## WEC Device Modeling

### Device Types
- **Point Absorbers**: Single-body oscillating devices (e.g., RM3)
- **Oscillating Wave Surge Converters**: Bottom-fixed flap-type devices
- **Oscillating Water Columns**: Air-filled chamber devices
- **Custom Models**: User-defined WEC characteristics

### Power Models
- **Simplified Models**: Power matrix-based generation
- **Detailed Models**: Integration with WECSim for high-fidelity simulation
- **Probabilistic Models**: Uncertainty quantification in power output

## WEC Farm Configuration

```python
from wecgrid.wec import WECFarm, WECDevice

# Create WEC farm
farm = WECFarm(name="North Coast Farm")

# Add individual devices
device1 = WECDevice(type="RM3", location=(0, 0))
device2 = WECDevice(type="RM3", location=(100, 0))

farm.add_device(device1)
farm.add_device(device2)

# Set environmental conditions
farm.set_wave_conditions(Hs=2.5, Tp=8.0, direction=0)

# Calculate power output
power_output = farm.calculate_power()
```

## Grid Integration

WEC farms connect to the power system through:

```python
import wecgrid

engine = wecgrid.Engine()
engine.case("IEEE_30_bus")

# Create and connect WEC farm
wec_farm = engine.create_wec_farm()
wec_farm.connect_to_bus(bus_number=15)

# Set WEC characteristics
wec_farm.set_rated_power(50)  # MW
wec_farm.set_capacity_factor(0.35)

# Run simulation
results = engine.run_simulation()
```

## Array Effects

WEC farms consider array interactions:

- **Wake Effects**: Downstream wave reduction
- **Near-Field Effects**: Wave interference between devices
- **Electrical Collection**: Submarine cable modeling
- **Maintenance Access**: Availability considerations

## Economic Modeling

```python
# Set economic parameters
wec_farm.set_economics({
    'capex': 5000,  # $/kW
    'opex': 150,    # $/kW/year
    'lifetime': 25, # years
    'discount_rate': 0.08
})

# Calculate metrics
lcoe = wec_farm.calculate_lcoe()
npv = wec_farm.calculate_npv(electricity_price=0.10)
```

## API Reference

![mkapi](wecgrid.wec.wecfarm.WECFarm)

![mkapi](wecgrid.wec.wecdevice.WECDevice)