# Installation

## System Requirements
- **Python**: 3.7 + 
- **Operating System**: Windows recommended for full functionality (PSS®E is Windows-only). Core features compatiable with most platforms.

### Power System Software
- **PSS®E**: Version 34 or later (commercial license required)
- **PyPSA**: PyPSA link here

### WEC Modeling Software
- **MATLAB**: R2021b + 
- **WEC-Sim**: Install separately ([WEC-Sim Installation Guide](https://wec-sim.github.io/WEC-Sim/main/user/getting_started.html))

<!-- ## WEC-Grid Installation

```bash
# Clone the repository
git clone https://github.com/acep-uaf/WEC-GRID.git
cd WEC-GRID

# Create and activate a conda environment
conda create -n wecgrid python=3.7
conda activate wecgrid

# Install in development mode
pip install -e .
```

## MATLAB Configuration
If using WEC-Sim:

1. Install MATLAB 2021b.
2. Install the MATLAB Engine API for Python:

```bash
cd "matlabroot/extern/engines/python"
python setup.py install
```

## PSS®E Configuration
If using the PSS®E backend:

1. Verify PSS®E installation:

```python
import psse35  # or your installed version
```

2. Add PSS®E to the Python path:

```python
import sys
sys.path.append(r'C:\\Program Files (x86)\\PTI\\PSSEXplore35\\PSSPY37')
```

## Verification

```python
import wecgrid
print(f"WEC-Grid version: {wecgrid.__version__}")

# Basic functionality
engine = wecgrid.Engine()
print("WEC-Grid installed successfully!")
```

## Next Steps

Once installed, head to the [Quick Start Guide](quickstart.md) for your first simulation.
 -->
