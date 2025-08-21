# Quick Start

This guide will walk you through your first WEC-Grid simulation in just a few minutes.

### Prerequisites
- WEC-Grid installed (see [Installation Guide](install.md))


### Step 1: Import WEC-Grid

```python
import wecgrid
```

### Step 2: Create a Basic Configuration

```python
# Initialize the engine
engine = wecgrid.Engine()

# Load a power system modeler
engine.load(["psse", "pypsa"]) 

# Initialize WEC-Grid with IEEE 14-bus system
engine.case("IEEE_14_bus")
```

### Step 3: Add a WEC Device

```python
engine.apply_wec(
    farm_name="WEC-FARM", 
    size=1, 
    sim_id=1, 
    bus_location=15, 
    connecting_bus=1, 
    model="RM3"
)
```
### Step 4: Run the Simulation

```python
engine.simulate()
```

### Step 5: View Results

```python
# Plot basic results
engine.plot.quick_overview()
engine.plot.plot_wec_analysis()

```

## Complete Example Script

Here's the full script you can copy and run:

```python
"""
WEC-Grid Quick Start Example
============================
A simple co-simulation of a wave energy converter with an IEEE test system.
"""

import wecgrid

def main():
    # Initialize the engine
    engine = wecgrid.Engine()

    # Load a power system modeler
    engine.load(["psse", "pypsa"]) 

    # Initialize WEC-Grid with IEEE 14-bus system
    engine.case("IEEE_14_bus")

    # Apply WEC devices
    engine.apply_wec(farm_name="WEC-FARM", size=1, sim_id=1, bus_location=15, connecting_bus=1, model="RM3")

    # Run the simulation
    engine.simulate()

    # Plot results
    engine.plot.quick_overview()
    engine.plot.plot_wec_analysis()

if __name__ == "__main__":
    main()
```
