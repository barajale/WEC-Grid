# Quick Start

This guide will walk you through your first WEC-Grid simulation in just a few minutes.

## Prerequisites

- WEC-Grid installed (see [Installation Guide](install.md))
- Python 3.8+ with basic packages (numpy, pandas, matplotlib)
- Optional: MATLAB with WEC-Sim for full functionality

## Your First Simulation

### Step 1: Import WEC-Grid

```python
import wecgrid
from wecgrid import WECGrid
from datetime import datetime
```

### Step 2: Create a Basic Configuration

```python
# Initialize WEC-Grid with IEEE 14-bus system
wec_grid = WECGrid(
    grid_model="IEEE_14_bus",
    power_system_backend="pypsa",  # or "psse" if available
    start_time=datetime(2023, 1, 1, 0, 0, 0),
    simulation_length=24,  # hours
    time_step="1H"  # hourly time steps
)
```

### Step 3: Add a WEC Device

```python
# Add an RM3 wave energy converter
from wecgrid.wec import WECDevice

wec_device = WECDevice(
    name="offshore_wec_1",
    model="RM3",
    bus_connection=14,  # Connect to bus 14
    capacity=1.0,  # 1 MW capacity
    location="offshore_site"
)

# Add device to the grid
wec_grid.add_wec_device(wec_device)
```

### Step 4: Run the Simulation

```python
# Execute the co-simulation
results = wec_grid.run_simulation()
print("Simulation completed successfully!")
```

### Step 5: View Results

```python
# Plot basic results
wec_grid.plot_summary()

# Access specific data
power_output = results.get_wec_power_output("offshore_wec_1")
grid_frequency = results.get_system_frequency()

print(f"Average WEC power: {power_output.mean():.2f} MW")
print(f"Grid frequency range: {grid_frequency.min():.3f} - {grid_frequency.max():.3f} Hz")
```

## Complete Example Script

Here's the full script you can copy and run:

```python
#!/usr/bin/env python3
"""
WEC-Grid Quick Start Example
============================
A simple co-simulation of a wave energy converter with an IEEE test system.
"""

import wecgrid
from wecgrid import WECGrid
from wecgrid.wec import WECDevice
from datetime import datetime

def main():
    print("WEC-Grid Quick Start Example")
    print("=" * 30)
    
    # Step 1: Initialize the grid model
    print("1. Setting up grid model...")
    wec_grid = WECGrid(
        grid_model="IEEE_14_bus",
        power_system_backend="pypsa",
        start_time=datetime(2023, 1, 1, 0, 0, 0),
        simulation_length=24,
        time_step="1H"
    )
    
    # Step 2: Add WEC device
    print("2. Adding WEC device...")
    wec_device = WECDevice(
        name="offshore_wec_1",
        model="RM3",
        bus_connection=14,
        capacity=1.0,
        location="offshore_site"
    )
    wec_grid.add_wec_device(wec_device)
    
    # Step 3: Run simulation
    print("3. Running co-simulation...")
    results = wec_grid.run_simulation()
    
    # Step 4: Display results
    print("4. Processing results...")
    power_output = results.get_wec_power_output("offshore_wec_1")
    print(f"   Average WEC power: {power_output.mean():.2f} MW")
    print(f"   Peak WEC power: {power_output.max():.2f} MW")
    
    # Step 5: Generate plots
    print("5. Generating plots...")
    wec_grid.plot_summary()
    
    print("\n✅ Quick start completed successfully!")
    print("Next steps:")
    print("  - Explore the Examples section for more complex scenarios")
    print("  - Check the User Guide for detailed component documentation")
    print("  - Try different grid models and WEC configurations")

if __name__ == "__main__":
    main()
```

## Understanding the Results

The simulation produces several key outputs:

- **WEC Power Output**: Time series of electrical power generation
- **Grid Metrics**: Voltage, frequency, and stability indicators  
- **System Loading**: Impact on transmission lines and transformers
- **Economic Data**: Generation costs and revenue (if configured)

## Next Steps

### Explore More Features

- **[Multiple WEC devices](examples/basic-example.md)**: Add WEC farms and arrays
- **[Different grid models](user-guide/models/grid-models.md)**: Try IEEE 30-bus or 39-bus systems
- **[Advanced configuration](user-guide/components/engine.md)**: Customize simulation parameters

### Learn the Components

- **[Engine](user-guide/components/engine.md)**: Core simulation coordinator
- **[Database](user-guide/components/database.md)**: Result storage and retrieval
- **[Plotting](user-guide/components/plotting.md)**: Visualization capabilities

### Common Next Steps

1. **Experiment with parameters**: Change WEC capacity, location, or grid model
2. **Add multiple devices**: Create WEC farms with different configurations
3. **Analyze results**: Use the database API to perform custom analysis
4. **Customize plots**: Create publication-ready figures

## Troubleshooting

If you encounter issues:

1. **Check installation**: Verify all dependencies are installed correctly
2. **Review error messages**: Most errors provide helpful debugging information
3. **Consult documentation**: See the [Troubleshooting Guide](reference/troubleshooting.md)
4. **Get help**: Post issues on [GitHub](https://github.com/acep-uaf/WEC-GRID/issues)

## Performance Tips

- Start with small simulations (24 hours, hourly steps)
- Use PyPSA backend for faster setup and testing
- Enable database storage for large parameter sweeps
- Consider parallel execution for multiple scenarios
