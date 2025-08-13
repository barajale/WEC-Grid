# Plotting & Visualization

The WEC-Grid plotting module provides comprehensive visualization capabilities for simulation results, grid topologies, and time-series data.

## Overview

The plotting system offers:

- **Grid Visualization**: Network topology plots with customizable styling
- **Time-Series Plots**: Power generation, loads, and WEC output over time
- **Result Analysis**: Power flow visualization and comparative analysis
- **Interactive Features**: Matplotlib-based interactive plots
- **Export Options**: High-quality figure export in multiple formats

## Plot Types

### Grid Topology Plots
- Bus layouts with voltage levels
- Transmission line representations
- Generator and load locations
- WEC farm positioning

### Time-Series Visualizations
- Power generation profiles
- WEC output characteristics
- Load demand patterns
- Grid frequency and voltage

### Analysis Plots
- Power flow distributions
- Voltage profiles
- Loss analysis
- Economic metrics

## Basic Usage

```python
from wecgrid.plot import WECGridPlotter

# Initialize plotter
plotter = WECGridPlotter()

# Plot grid topology
plotter.plot_network(network_data)

# Plot time series results
plotter.plot_time_series(time_data, power_data)

# Create power flow visualization
plotter.plot_power_flow(pf_results)

# Export plots
plotter.save_figure("output.png", dpi=300)
```

## Integration with Engine

Plotting is integrated with the main simulation workflow:

```python
import wecgrid

engine = wecgrid.Engine()
results = engine.run_simulation()

# Plot results directly
engine.plot_results()

# Or use dedicated plotting
plotter = engine.get_plotter()
plotter.plot_network()
```

## Customization

The plotting module supports extensive customization:

```python
# Custom styling
plotter.set_style({
    'grid_color': 'blue',
    'bus_size': 50,
    'line_width': 2
})

# Multiple subplots
fig, axes = plotter.create_subplots(2, 2)
plotter.plot_voltage_profile(ax=axes[0,0])
plotter.plot_power_flow(ax=axes[0,1])
```

## API Reference

![mkapi](wecgrid.plot.wecgrid_plotter.WECGridPlotter)