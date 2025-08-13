# WEC-GRID Documentation Style Guide

This guide establishes consistent documentation standards for the WEC-GRID package, based on Google docstring style with domain-specific extensions for power systems and WEC modeling.

## General Principles

- **Concise but Complete**: Provide essential information without excessive detail
- **User-Focused**: Write for researchers and engineers using the software
- **Consistent Formatting**: Follow the patterns below for all code
- **Technical Accuracy**: Include units, constraints, and API-specific notes

## Class Documentation

### Standard Class Docstring Structure

```python
class ClassName:
    """Brief one-line description of the class purpose.
    
    More detailed description explaining the class's role in the WEC-GRID framework.
    Focus on what it does, why it's needed, and how it fits into the bigger picture.
    
    Args:
        param1 (type): Description of parameter.
        param2 (type, optional): Description with default behavior. Defaults to value.
        
    Attributes:
        attr1 (type): Description of public attribute.
        attr2 (type): Description of another attribute.
        
    Example:
        >>> # Minimal working example
        >>> obj = ClassName(param1="value")
        >>> result = obj.method()
        >>> print(f"Result: {result}")
        
    Notes:
        - Important implementation details
        - API-specific information (PSS®E units, PyPSA conventions)
        - Cross-platform compatibility notes
        
    TODO:
        - List of planned improvements
        - Known limitations to address
    """
```

### Attributes Section Guidelines

- **Public attributes only**: Don't document private (`_attr`) attributes
- **Include units**: Specify [MW], [p.u.], [degrees], etc. where applicable
- **Type hints**: Use proper type annotations in the signature

## Method Documentation

### Standard Method Docstring Structure

```python
def method_name(self, param1: type, param2: type = default) -> return_type:
    """Brief description of what the method does.
    
    Optional longer description for complex methods. Explain the algorithm,
    data flow, or important behavioral details.
    
    Args:
        param1 (type): Parameter description with units if applicable.
        param2 (type, optional): Description. Defaults to default.
        
    Returns:
        return_type: Description of return value with units/format.
        
    Raises:
        SpecificError: When this error occurs.
        AnotherError: When this other error occurs.
        
    Example:
        >>> result = obj.method_name("input", param2=42)
        >>> print(f"Output: {result}")
        
    Notes:
        - PSS®E API specific: Values in system base MVA
        - PyPSA convention: Per-unit on 100 MVA base
        - Cross-platform validation tested
        
    TODO:
        - Performance optimization needed
        - Add support for additional case formats
    """
```

## Function Documentation

### Standalone Functions

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Brief description of function purpose.
    
    Args:
        param1 (type): Description.
        param2 (type, optional): Description. Defaults to default.
        
    Returns:
        return_type: Description.
        
    Example:
        >>> result = function_name("input")
        
    Notes:
        - Domain-specific notes
        - Units and conventions
    """
```

## Dataclass Documentation

```python
@dataclass
class DataClassName:
    """Brief description of the data structure.
    
    Detailed description of what this dataclass represents and how it's used
    within the WEC-GRID framework.
    
    Attributes:
        field1 (type): Description with units [MW] if applicable.
        field2 (type): Description of the field's purpose.
        field3 (type, optional): Optional field description.
        
    Example:
        >>> data = DataClassName(field1=100.0, field2="value")
        >>> print(f"Field1: {data.field1} MW")
        
    Notes:
        - Data validation rules
        - Expected ranges or constraints
        
    TODO:
        - Add validation methods
    """
    field1: float
    field2: str
    field3: Optional[str] = None
```

## Section Guidelines

### Args Section
- **Order**: Required parameters first, then optional
- **Types**: Always include type hints in both signature and docstring
- **Units**: Specify units for physical quantities: `[MW]`, `[p.u.]`, `[degrees]`, `[seconds]`
- **Constraints**: Mention valid ranges or constraints where relevant

### Returns Section
- **Type**: Always specify return type
- **Format**: Describe data structure for complex returns
- **Units**: Include units for physical quantities

### Example Section
- **Minimal working example**: Show basic usage
- **Real values**: Use realistic parameter values
- **Output**: Show expected output when helpful
- **Comments**: Explain non-obvious steps

### Notes Section
Use for:
- **API specifics**: "PSS®E values in system MVA base", "PyPSA per-unit on 100 MVA"
- **Cross-platform differences**: "PyPSA uses different sign convention"
- **Performance considerations**: "Large networks may require chunking"
- **Validation status**: "Cross-platform validation completed"
- **Usage warnings**: "Requires PSS®E license", "MATLAB required for WEC-Sim"

### TODO Section
Use for:
- **Known limitations**: "Limited to 9 WEC farms due to PSS®E ID constraints"
- **Planned features**: "Add support for WECC models"
- **Performance improvements**: "Optimize for large datasets"
- **API changes**: "Replace deprecated PSS®E calls"

## Domain-Specific Guidelines

### Power System Conventions
- **Base values**: Always specify MVA base (typically 100 MVA)
- **Units**: Use standard power engineering units
  - Power: `[MW]`, `[MVAr]`, `[MVA]`
  - Voltage: `[kV]`, `[p.u.]`
  - Impedance: `[p.u.]`, `[ohms]`
  - Angle: `[degrees]`, `[radians]`
  
### WEC-Specific Terms
- **Power output**: Always specify if electrical or mechanical
- **Time series**: Mention sampling rate and duration
- **Wave conditions**: Include significant wave height, period
- **Device models**: Reference standard models (RM3, OSWEC, etc.)

### Software Integration
- **PSS®E**: Mention version compatibility, license requirements
- **PyPSA**: Note open-source nature, installation requirements
- **WEC-Sim**: Specify MATLAB dependency, tested versions
- **Cross-platform**: Highlight validation between tools

## Examples of Concise vs. Verbose

### ❌ Too Verbose
```python
def load(self, software: List[str]) -> None:
    """Initialize power system simulation backends for comprehensive analysis.
    
    This method performs the critical task of initializing one or more power 
    system modeling platforms using the previously loaded case file through 
    the case() method. Each software backend provides completely independent 
    simulation capabilities and can be utilized for extensive cross-platform 
    validation studies to ensure accuracy and reliability across different 
    modeling approaches and methodologies in power system analysis.
    
    The method supports both commercial and open-source platforms, enabling
    researchers to leverage the strengths of each tool while maintaining
    consistency in simulation setup and execution...
    """
```

### ✅ Concise and Clear
```python
def load(self, software: List[str]) -> None:
    """Initialize power system modeling backends.
    
    Args:
        software (List[str]): Backends to load ("psse", "pypsa").
        
    Raises:
        ValueError: If no case file loaded or invalid software name.
        RuntimeError: If initialization fails (missing license, etc.).
        
    Example:
        >>> engine.case("IEEE_30_bus")
        >>> engine.load(["psse", "pypsa"])
        
    Notes:
        - PSS®E requires commercial license
        - PyPSA is open-source
        - Enables cross-platform validation
        
    TODO:
        - Add error handling for license failures
    """
```

## Module-Level Documentation

```python
"""Brief module description.

Longer description of module purpose and contents. Explain how this module
fits into the WEC-GRID framework.

Example:
    >>> from wecgrid.modelers import PSSEModeler
    >>> modeler = PSSEModeler(engine)

Notes:
    - Module-specific requirements or dependencies
    - Important usage patterns
"""
```

## Summary

This style guide prioritizes:
1. **Clarity over completeness**: Essential information without bloat
2. **Consistency**: Same format across all code
3. **Domain awareness**: Power system and WEC-specific context
4. **User focus**: Written for package users, not implementers
5. **Actionable TODOs**: Clear development priorities

Follow this guide to maintain professional, consistent documentation throughout the WEC-GRID package.
