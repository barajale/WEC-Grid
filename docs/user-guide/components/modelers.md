# Power System Modelers

The modelers module provides different backends for power system analysis, allowing users to choose the most appropriate tool for their specific needs.

## Overview

WEC-Grid supports multiple power system modeling backends:

- **PSS/E Modeler**: Industry-standard commercial power system analysis
- **PyPSA Modeler**: Open-source Python-based power system optimization
- **Grid State Manager**: Common interface for grid model management

## Architecture

The modeler architecture provides:

- **Unified Interface**: Common API across different backends
- **Backend Selection**: Choose the most appropriate tool for your analysis
- **Model Translation**: Convert between different model formats
- **Result Standardization**: Consistent output format regardless of backend

## Supported Backends

### PSS/E (Siemens)
- Industry-standard power flow and dynamic analysis
- Large-scale transmission system modeling
- Comprehensive stability analysis capabilities
- Commercial license required

### PyPSA (Open Source)
- Modern Python-based optimization framework
- Multi-period planning and dispatch optimization
- Open-source with active development community
- Excellent for renewable energy integration studies

## Backend Selection

```python
import wecgrid

# Initialize with specific backend
engine = wecgrid.Engine()
engine.load(["psse"])    # Use PSS/E backend
# or
engine.load(["pypsa"])   # Use PyPSA backend

# Or let WEC-Grid choose automatically
engine.load_auto()  # Selects available backend
```

## Model Management

The Grid State Manager provides common functionality:

```python
from wecgrid.modelers import GridState

# Load a grid model
grid_state = GridState()
grid_state.load_model("IEEE_30_bus.RAW")

# Access model components
buses = grid_state.get_buses()
branches = grid_state.get_branches()
generators = grid_state.get_generators()

# Modify the model
grid_state.add_bus(bus_id=31, voltage=138.0)
grid_state.add_generator(bus_id=31, capacity=100.0)
```

## Cross-Backend Compatibility

Models can be converted between backends:

```python
# Load in PSS/E format
psse_model = engine.load_psse_case("model.RAW")

# Convert to PyPSA
pypsa_model = engine.convert_to_pypsa(psse_model)

# Or use the same model with different backends
engine.switch_backend("pypsa")
results_pypsa = engine.run_powerflow()

engine.switch_backend("psse")
results_psse = engine.run_powerflow()
```

## Analysis Capabilities

Different backends offer different analysis types:

### Common Analysis
- Power flow (Newton-Raphson, Fast-Decoupled)
- Contingency analysis
- Voltage stability assessment

### PSS/E Specific
- Dynamic simulation
- Short-circuit analysis
- Protection coordination

### PyPSA Specific
- Unit commitment optimization
- Transmission expansion planning
- Multi-period dispatch

## API Reference

![mkapi](wecgrid.modelers.power_system_modeler.PowerSystemModeler)

![mkapi](wecgrid.modelers.grid_state.GridState)