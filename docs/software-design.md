# Software Design

WEC-Grid follows a modular architecture with clear separation between marine energy and power system domains.

## System Architecture

### Basic Overview

<img src="../diagrams/uml/WEC_Grid_uml.png" alt="System Architecture Overview" width="600"/>

<img src="../diagrams/workflow/WEC_Grid_workflow.png" alt="Flowchart" width="600"/>

<img src="../diagrams/sequence/WEC_Grid_sequence.png" alt="Sequence Diagram" width="600"/>
 

## Design Principles

- **Bridge Pattern**: Unified interface across different power system backends
- **Modularity**: Clear separation between WEC modeling and power system analysis
- **Extensibility**: Easy integration of new models and backends
- **Data Persistence**: Centralized SQLite database for reproducible research


### Engine
Central coordinator managing simulation workflow and component interaction.

### Power System Modelers  
Standardized interface for PSS®E and PyPSA backends with consistent data formats.

### WEC Modeling
Device and farm-level modeling with integration to WEC-Sim for high-fidelity simulation.

### Data Management
SQLite database with structured storage for simulation metadata and time-series results.

## Workflow

[Include your flowchart and describe the typical simulation workflow]
