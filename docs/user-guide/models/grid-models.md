# Grid Models

WEC-Grid includes several IEEE standard test systems that provide well-characterized power grid models for research and validation.

## Available Models

### IEEE 14-Bus System
- **Description**: Small test system with 14 buses, 5 generators, and 11 loads
- **Use case**: Initial testing and validation of small-scale integrations
- **Characteristics**: Simple radial and loop structures
- **Citation**: [IEEE 14-bus test case reference]

### IEEE 24-Bus System  
- **Description**: Reliability test system with 24 buses
- **Use case**: Medium-scale reliability and planning studies
- **Characteristics**: Multiple voltage levels, generator diversity
- **Citation**: [IEEE 24-bus reliability test system reference]

### IEEE 30-Bus System
- **Description**: Standard test case with 30 buses, 6 generators
- **Use case**: Power flow and optimization studies  
- **Characteristics**: Well-balanced load and generation
- **Citation**: [IEEE 30-bus test case reference]

### IEEE 39-Bus System
- **Description**: New England test system with 39 buses, 10 generators
- **Use case**: Large-scale stability and dynamic studies
- **Characteristics**: Complex interconnected system
- **Citation**: [IEEE 39-bus New England system reference]

## Model Format

All grid models are provided in PSS®E .RAW format and can be automatically converted for use with PyPSA. The models include:

- Bus data (voltage levels, load specifications)
- Branch data (transmission lines, transformers)
- Generator data (capacity, costs, constraints)
- Load data (demand profiles, power factors)

## Usage

Grid models are automatically loaded by WEC-Grid based on the specified system:

```python
from wecgrid import WECGrid

# Load IEEE 14-bus system
wec_grid = WECGrid(grid_model="IEEE_14_bus")
```

## Modifications

The standard IEEE models have been adapted for WEC integration studies:

- Added appropriate bus locations for WEC connections
- Included representative load profiles for coastal regions
- Maintained original system characteristics for validation

## References

[Include proper citations for each IEEE test system]
