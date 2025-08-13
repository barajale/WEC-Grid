# PSS/E Modeler

The PSS/E Modeler provides integration with Siemens PSS®E for power system analysis within WEC-Grid simulations.

## Overview

The `PSSEModeler` class serves as a bridge between WEC-Grid and PSS®E, enabling:

- Power flow analysis using PSS®E solvers
- Grid model import from PSS®E .RAW files
- WEC integration into existing PSS®E models
- Result extraction and standardization

## Requirements

- PSS®E version 34 or later
- Valid PSS®E license
- PSS®E Python API properly installed

## Basic Usage

```python
from wecgrid.modelers import PSSEModeler

# Initialize PSS/E modeler
psse_modeler = PSSEModeler()

# Load a grid model
psse_modeler.load_case("IEEE_30_bus.RAW")

# Run power flow
results = psse_modeler.run_powerflow()
```

## Integration with Engine

The PSS/E modeler is typically used through the main Engine:

```python
import wecgrid

engine = wecgrid.Engine()
engine.case("IEEE_30_bus")
engine.load(["psse"])  # Load PSS/E backend

# Now you can run simulations with PSS/E
```

## API Reference

![mkapi](wecgrid.modelers.psse_modeler.PSSEModeler)