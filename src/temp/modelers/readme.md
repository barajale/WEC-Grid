# NetworkState Data Format

This document describes the standardized snapshot and time-series structure used by all power system modelers in the WEC-Grid framework. Each modeler (e.g., PSSE, PyPSA) must populate the `NetworkState` object with a consistent set of DataFrames.

Each component (bus, gen, load, line) is stored in two parts:

- **Snapshot**: The current state as a `pd.DataFrame` (e.g., `state.bus`)
- **Time-Series**: A dictionary of DataFrames indexed by timestamp (e.g., `state.bus_t["p"]`)

> 🧠 **Note**: All values are reported in **per-unit (pu)** unless otherwise noted.  
> 🔢 **Important**: The index of each DataFrame should be a simple integer or timestamp index—not a repeated ID column.

---

## DataFrame Specifications

### `bus` DataFrame

| Column      | Description                                      | Type   |
|-------------|--------------------------------------------------|--------|
| `bus`       | Bus number of the corresponding node             | int    |
| `bus_name`  | Optional name of the bus                         | str    |
| `type`      | Bus type: 3 = swing, 2 = PV, 1 = PQ               | int    |
| `p`         | Active power injection at the bus (pu)           | float  |
| `q`         | Reactive power injection at the bus (pu)         | float  |
| `v_mag`     | Voltage magnitude (pu)                           | float  |
| `angle_deg` | Voltage angle in degrees                         | float  |
| `base`      | MVA base of the bus                              | float  |

---

### `gen` DataFrame

| Column   | Description                                   | Type   |
|----------|-----------------------------------------------|--------|
| `gen`    | Generator ID (e.g., `"G0"`, `"G1"`)            | str    |
| `bus`    | Associated bus number                         | int    |
| `p`      | Active power output (pu)                      | float  |
| `q`      | Reactive power output (pu)                    | float  |
| `base`   | MVA base of the generator                     | float  |
| `status` | Generator status: 1 = online, 0 = offline     | int    |

---

### `line` DataFrame

| Column     | Description                                                               | Type   |
|------------|---------------------------------------------------------------------------|--------|
| `line`     | Line ID (e.g., `"Line_1_2_0"`)                                             | str    |
| `ibus`     | From-bus number                                                           | int    |
| `jbus`     | To-bus number                                                             | int    |
| `line_pct` | Line loading as a percentage of rated capacity (**not** in pu)            | float  |
| `status`   | Line status: 1 = online, 0 = offline                                      | int    |

---

### `load` DataFrame

| Column      | Description                                                               | Type   |
|-------------|---------------------------------------------------------------------------|--------|
| `load`      | Load ID (e.g., `"Load_1_0"`)                                               | str    |
| `bus`       | Associated bus number                                                     | int    |
| `p`         | Active power consumption (pu)                                             | float  |
| `q`         | Reactive power consumption (pu)                                           | float  |
| `base`      | MVA base of the load                                                      | float  |
| `status`    | Load status: 1 = online, 0 = offline                                      | int    |

---

## Field Descriptions

| Field       | Description                                                                                   | Type   |
|-------------|-----------------------------------------------------------------------------------------------|--------|
| `bus`       | Bus number in the grid                                                                        | int    |
| `ibus`      | "From" bus in a transmission line                                                             | int    |
| `jbus`      | "To" bus in a transmission line                                                               | int    |
| `v_mag`     | Voltage magnitude (pu)                                                                        | float  |
| `bus_name`  | Human-readable bus label                                                                      | str    |
| `type`      | Bus type: 3 = swing, 2 = PV, 1 = PQ                                                            | int    |
| `gen`       | Generator ID (e.g., `"G0"`)                                                                   | str    |
| `load`      | Load ID (e.g., `"Load_5_1"`)                                                                   | str    |
| `line`      | Line ID (e.g., `"Line_2_4_0"`)                                                                 | str    |
| `status`    | Operational status: 1 = online, 0 = offline                                                   | int    |
| `p`         | Active power (pu)                                                                             | float  |
| `q`         | Reactive power (pu)                                                                           | float  |
| `base`      | MVA base for power normalization                                                              | float  |
| `angle_deg` | Voltage angle in degrees                                                                      | float  |
| `line_pct`  | Line loading as percent of the thermal limit (e.g., 78.5 = 78.5% loaded)                      | float  |

---

## Notes for Developers

- Always include `df.attrs["df_type"]` when updating time-series using `state.update(component, timestamp, df)`.
- Do **not** set the DataFrame index to `bus`, `gen`, `line`, or `load`. These should be columns only.
- All modelers must conform to these schema definitions to ensure compatibility across simulations and visualizations.