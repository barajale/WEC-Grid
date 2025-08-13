# Installation

## System Requirements

WEC-Grid requires:

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **MATLAB**: R2020b or later (for WEC-Sim integration)
- **Memory**: 4GB RAM minimum (8GB recommended)

### Power System Software (Optional)

For full functionality, install one or both:

- **PSS®E**: Version 34 or later (commercial license required)
- **PyPSA**: Installed automatically with WEC-Grid

## Installation Methods

### Option 1: pip install (Recommended)

```bash
pip install wecgrid
```

### Option 2: Development Installation

For developers or to access the latest features:

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

### Option 3: Conda Environment (Recommended for Research)

Use the provided environment file for a complete setup:

```bash
# Download the environment file
curl -O https://raw.githubusercontent.com/acep-uaf/WEC-GRID/main/wec_grid_env.yml

# Create and activate environment
conda env create -f wec_grid_env.yml
conda activate wecgrid

# Install WEC-Grid
pip install wecgrid
```

## MATLAB Configuration

### WEC-Sim Setup

1. **Install WEC-Sim**:
   ```matlab
   % In MATLAB, add WEC-Sim to path
   addpath(genpath('path/to/WEC-Sim'))
   ```

2. **Verify MATLAB Engine**:
   ```bash
   python -c "import matlab.engine; print('MATLAB Engine installed successfully')"
   ```

3. **Install MATLAB Engine for Python** (if needed):
   ```bash
   # Navigate to MATLAB installation directory
   cd "matlabroot/extern/engines/python"
   python setup.py install
   ```

## PSS®E Configuration (Optional)

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

## Troubleshooting

### Common Issues

**MATLAB Engine Not Found**:
```bash
# Reinstall MATLAB Engine
cd "matlabroot/extern/engines/python"
python setup.py install --force
```

**PSS®E Import Error**:
- Verify PSS®E installation and licensing
- Check Python version compatibility with PSS®E

**Permission Errors**:
```bash
# Use virtual environment
python -m venv wecgrid_env
source wecgrid_env/bin/activate  # Linux/Mac
# or
wecgrid_env\Scripts\activate  # Windows
pip install wecgrid
```

### Getting Help

- **Documentation**: [troubleshooting guide](reference/troubleshooting.md)
- **Issues**: [GitHub Issues](https://github.com/acep-uaf/WEC-GRID/issues)
- **Contact**: [Research team contact information]

## Next Steps

Once installed, proceed to the [Quick Start Guide](quickstart.md) for your first simulation.
