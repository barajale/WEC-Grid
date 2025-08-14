# WEC-Grid

**WEC-Grid** is an open-source software framework that integrates Wave Energy Converter (WEC) models into power system simulations, enabling preliminary integration studies of renewable wave energy with grid steady-state analysis.

## Overview

Wave energy integration into power grids faces a critical modeling gap: current power system tools (such as PSS®E, PyPSA) and marine hydrodynamic simulators (WEC-Sim) operate independently, hampering collaboration between marine energy and power system communities.

WEC-Grid bridges this gap by providing a unified modeling approach that accurately represents interactions between hydrodynamic behaviors and electrical power systems.

## Key Capabilities

- **Multi-Platform Integration**: Seamlessly coordinates PSS®E, PyPSA, and WEC-Sim simulations
- **Standardized Workflows**: Consistent API across different power system backends
- **Quasi-Steady-State Analysis**: Efficient power flow and voltage stability studies with WEC integration
- **Data Management**: SQLite database with standardized result formats for reproducible research
- **Visualization**: Time-series plots, cross-platform comparisons, and network diagrams

## Research Applications

WEC-Grid enables researchers to:

- Evaluate WEC performance under varying grid and sea state conditions
- Analyze WEC impacts on grid stability, reliability, and power quality 
- Conduct comparative studies across different simulation platforms

## Software Architecture

The framework employs a modular bridge pattern with:

- **Engine**: Central coordinator managing simulation timing, data flow, and result collection
- **Software Wrappers**: Standardized interfaces for PSS®E, PyPSA, and WEC-Sim
- **Database**: Persistent storage for simulation metadata and results
- **Visualization**: Integrated plotting capabilities for analysis and reporting


## Getting Started

- **[Installation](install.md)** - Setup instructions and dependencies
- **[Quick Start](quickstart.md)** - Your first WEC-Grid simulation
- **[Examples](examples/basic-example.md)** - Complete workflow demonstrations

## Support

- **Documentation**: [acep-uaf.github.io/WEC-Grid](https://acep-uaf.github.io/WEC-Grid/)
- **Repository**: [github.com/acep-uaf/WEC-GRID](https://github.com/acep-uaf/WEC-GRID)
- **Contact**: barajale@oregonstate.edu

## Acknowledgments

This work is supported by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy, Water Power Technology Office (Grant #DE-EE0009445), University of Alaska Fairbanks, and Pacific Northwest National Laboratory.


## Future Work

- Fault Analysis
- PowerFactory API
- Dynamic Simulation