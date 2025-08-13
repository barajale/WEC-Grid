# WEC-Grid

**WEC-Grid** is a Python framework for integrating **Wave Energy Converter (WEC)** models with power system simulation tools. It bridges the gap between device-level hydrodynamic modeling and grid-scale power system analysis.

## What is WEC-Grid?

WEC-Grid enables researchers and engineers to:

- **Model wave energy systems** at both device and grid scales
- **Integrate WEC farms** into existing power system models
- **Analyze grid impacts** of wave energy deployment
- **Perform co-simulation** between WEC-Sim and power system tools

## Key Features

### 🌊 **Wave Energy Modeling**
- Integration with MATLAB WEC-Sim for device-level physics
- Support for validated WEC models (RM3, LUPA)
- Scalable from single devices to large arrays

### ⚡ **Power System Integration**
- Compatible with PSS®E and PyPSA backends
- IEEE standard test systems included
- Grid connection and control modeling

### 📊 **Data Management**
- SQLite database for simulation results
- Standardized data APIs for reproducible workflows
- Built-in visualization and plotting tools

### 🔧 **Workflow Support**
- Time-synchronized co-simulation
- Automated result collection and storage
- Extensible architecture for new models

## Why WEC-Grid?

Traditional wave energy studies often focus on either device-level performance or grid-level impacts in isolation. WEC-Grid provides the missing link, enabling:

- **Comprehensive analysis** of wave energy integration scenarios
- **Realistic modeling** of both resource variability and grid constraints  
- **Standardized workflows** for comparative studies
- **Open-source platform** for collaborative research

## Funding Acknowledgment

This work is supported by the Alaska Center for Energy and Power (ACEP) at the University of Alaska Fairbanks.

## Quick Links

- **[Installation Guide](install.md)** - Get started with WEC-Grid
- **[Quick Start Tutorial](quickstart.md)** - Your first simulation
- **[Examples](examples/basic-example.md)** - Complete workflow examples
- **[API Reference](reference/api.md)** - Detailed documentation

## Citation

If you use WEC-Grid in your research, please cite:

```bibtex
@software{wecgrid2025,
  title={WEC-Grid: Integrating Wave Energy Converter Models into Power System Simulations},
  author={Alexander Barajas-Ritchie},
  year={2025},
  url={https://github.com/acep-uaf/WEC-GRID},
  institution={Alaska Center for Energy and Power, University of Alaska Fairbanks}
}
```

## License

WEC-Grid is released under the MIT License. See [LICENSE](https://github.com/acep-uaf/WEC-GRID/blob/main/LICENSE) for details.
