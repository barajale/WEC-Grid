# PSS/E Modeler

Integration with Siemens PSS®E for industry-standard power system analysis.

## Requirements

- PSS®E version 34 or later
- Valid PSS®E license
- PSS®E Python API

## Basic Usage

```python
# Load PSS/E backend
engine.load(["psse"])

# PSS/E automatically handles:
# - RAW file import
# - Power flow calculations  
# - WEC integration as generators
# - Result extraction
```

## API Reference

![mkapi](wecgrid.modelers.psse_modeler.PSSEModeler)