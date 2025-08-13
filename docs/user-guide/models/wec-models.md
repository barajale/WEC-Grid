# WEC Models

WEC-Grid includes validated wave energy converter models from academic research, providing realistic device characteristics for integration studies.

## Available Models

### RM3 Reference Model
- **Description**: Two-body point absorber developed by Sandia National Laboratories
- **Type**: Point absorber with vertical motion
- **Power Rating**: 1 MW nominal
- **Applications**: Offshore deployment scenarios
- **Validation**: Extensively validated against experimental data
- **Citation**: [RM3 reference model citation]

**Key Characteristics:**
- Dual-body design with float and submerged reaction body  
- Power take-off through relative motion
- Representative of commercial point absorber designs
- Well-documented hydrodynamic coefficients

### LUPA Model
- **Description**: [Add description of LUPA model]
- **Type**: [Add type information]
- **Power Rating**: [Add power rating]
- **Applications**: [Add application scenarios]
- **Citation**: [LUPA model citation]

## Model Integration

WEC models in WEC-Grid provide:

- **Hydrodynamic modeling**: Integration with WEC-Sim for device-level physics
- **Power conversion**: Realistic power take-off system modeling
- **Grid interface**: Appropriate electrical characteristics for grid connection
- **Scalability**: Support for single devices and arrays

## Custom Modifications

The standard academic models have been enhanced for grid integration:

- **Electrical interface modeling**: Added grid-connection components
- **Control system integration**: Incorporated grid-friendly control strategies
- **Array modeling**: Support for multiple device deployments
- **Environmental coupling**: Integration with wave resource data

## Usage

WEC models are selected when configuring WEC devices:

```python
from wecgrid.wec import WECDevice

# Create RM3 device
wec_device = WECDevice(
    model="RM3",
    location="offshore_site_1",
    capacity=1.0  # MW
)
```

## Validation

All WEC models included in WEC-Grid have been validated against:

- Original experimental or numerical data
- Independent modeling results
- Field deployment data (where available)

## References

[Include proper citations for each WEC model and any custom modifications]
