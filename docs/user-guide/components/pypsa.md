# PyPSA Modeler

The PyPSA Modeler provides integration with PyPSA (Python for Power System Analysis) for modern power system modeling within WEC-Grid.

## Overview

The `PyPSAModeler` class integrates with the PyPSA library to provide:

- Open-source power system modeling
- Time-series optimization capabilities
- Multi-period planning studies
- Modern Python-based power flow analysis

## Features

- Network topology creation and modification
- Time-series load and generation profiles
- Optimal power flow (OPF) solving
- Integration with renewable energy sources
- Export capabilities to various formats

## Basic Usage

```python
from wecgrid.modelers import PyPSAModeler

# Initialize PyPSA modeler
pypsa_modeler = PyPSAModeler()

# Create or load a network
pypsa_modeler.create_network()

# Add components
pypsa_modeler.add_bus("Bus1", x=0, y=0)
pypsa_modeler.add_generator("Gen1", "Bus1", p_nom=100)

# Solve optimal power flow
results = pypsa_modeler.solve_opf()
```

## Integration with Engine

Use PyPSA through the main Engine interface:

```python
import wecgrid

engine = wecgrid.Engine()
engine.case("custom_network")
engine.load(["pypsa"])  # Load PyPSA backend

# Configure and run simulations
```

## API Reference

![mkapi](wecgrid.modelers.pypsa_modeler.PyPSAModeler)