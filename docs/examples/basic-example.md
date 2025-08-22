# Basic Example

This short script runs a minimal WEC-Grid simulation using the built-in IEEE 14-bus case.

```python
import wecgrid

def main():
    engine = wecgrid.Engine()
    engine.case("IEEE_14_bus")
    engine.load(["pypsa"])
    engine.apply_wec(
        farm_name="DemoFarm",
        size=1,
        wec_sim_id=1,
        bus_location=15,
        connecting_bus=1,
        scaling_factor=1,
    )
    engine.simulate()
    engine.plot.quick_overview()

if __name__ == "__main__":
    main()
```

For more details, see the [Quick Start guide](../quickstart.md).


