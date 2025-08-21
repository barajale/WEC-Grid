# Power System Modelers

Backend power system analysis tools providing unified interface across different solvers.

## Supported Backends

- **PSS®E**: Industry-standard commercial power system analysis
- **PyPSA**: Open-source Python-based optimization framework

## Basic Usage

```python
# Same API regardless of backend
engine.load(["psse"])    # or ["pypsa"] or both
engine.case("IEEE_30_bus")
engine.simulate()

# Cross-platform validation
engine.load(["psse", "pypsa"])
engine.simulate()
engine.compare_results()
```

## API Reference

![mkapi](wecgrid.modelers.power_system_modeler.PowerSystemModeler)

![mkapi](wecgrid.modelers.grid_state.GridState)