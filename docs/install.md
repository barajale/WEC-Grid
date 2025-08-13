# Installation

## System Requirements
- **Python**: 3.7
- **Operating System**: Windows


### Power System Software
- **PSS®E**: Version 34 or later (commercial license required)
- **PyPSA**: Installed automatically with WEC-Grid

### WEC Modeling Softwares
- **MATLAB**: R2021b 
- **WEC-Sim**: Installed separately (see [WEC-Sim Installation Guide](https://wec-sim.github.io/WEC-Sim/main/user/getting_started.html))

## WEC-Grid Installation

```bash
# Clone the repository
git clone https://github.com/acep-uaf/WEC-GRID.git
cd WEC-GRID

# Create conda environment
conda env create -f wec_grid_env.yml
conda activate wecgrid

# Install in development mode
pip install -e .
```



## MATLAB Configuration

### 1. Install MATLAB 2021b

[MATLAB Install](https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html#mw_911bcad0-9c6f-49cb-ae22-ca8a1b3ea29e)

### 2. Install MATLAB Engine API for Python

```bash
conda activate <environment_name>
# Navigate to MATLAB installation directory
cd "matlabroot/extern/engines/python"
python setup.py install
```

## PSS®E Configuration

If using PSS®E backend:

1. **Verify PSS®E Installation**:
```python
import psse35  # or your PSS®E version
print("PSS®E available")
```

2. **Add PSS®E to Python Path**:
```python
import sys
sys.path.append(r'C:\Program Files (x86)\PTI\PSSEXplore35\PSSPY37')
```
For additional help search the [PSS/E Python API Forum](https://psspy.org/psse-help-forum/questions/)

## Verification

Test your installation:

```python
import wecgrid
print(f"WEC-Grid version: {wecgrid.__version__}")

# Test basic functionality
from wecgrid import WECGrid
wec_grid = WECGrid()
print("WEC-Grid installed successfully!")
```

## Next Steps

Once installed, proceed to the [Quick Start Guide](quickstart.md) for your first simulation.
