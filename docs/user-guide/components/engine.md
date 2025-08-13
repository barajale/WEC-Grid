# Engine

The `Engine` is the main orchestrator for WEC‑Grid simulations. It wires together:
- **Power-system backends**: PSS®E (`PSSEModeler`) and/or PyPSA (`PyPSAModeler`)
- **WEC farm modeling** and optional WEC‑SIM runs (`WECFarm`, `WECSimRunner`)
- **Data persistence** (`WECGridDB`) and **plotting** (`WECGridPlotter`)
- **Time management** (`WECGridTimeManager`)

> Python support: `3.7` (pinned for this release)

---

## Quick start

```python
import wecgrid

# 1) Build the engine
engine = wecgrid.Engine()

# 2) Select a power-flow case
#    Accepts:
#    - Name of a bundled case (e.g., "IEEE_30_bus" or "IEEE_30_bus.RAW")
#    - A full local path to a .RAW file
engine.case("IEEE_30_bus")

# 3) Load one or both backends
engine.load(["psse", "pypsa"])   # or just ["pypsa"]

# 4) (Optional) Attach a WEC farm using results already in the database (sim_id),
#    or run WEC‑SIM first (see WECSim section below) and then attach.
engine.apply_wec(
    farm_name="CoastalFarm",
    size=8,
    sim_id=1,             # existing WECSim run stored in the DB (or -1 if none yet)
    model="RM3",          # built-in model name OR path to a WEC model folder
    bus_location=31,
    connecting_bus=1
)

# 5) Run the simulation
engine.simulate(load_curve=True, plot=True)
```

---

## Selecting a grid case

`Engine.case()` accepts:
- The **name** (or stem) of a *bundled* RAW file (e.g., `"IEEE_14_bus"`, `"IEEE_30_bus.RAW"`), or
- A **full local path** (`C:/…/my_system.RAW`).

Under the hood, the engine uses an internal resolver so you don’t need to manage package paths.

```python
engine.case("IEEE_30_bus")            # bundled
# or
engine.case(r"C:\data\grids\my.RAW")  # local
```

If the file is not found, a `FileNotFoundError` is raised.

---

## Loading backends

```python
engine.load(["psse", "pypsa"])  # initialize PSS®E and PyPSA
# engine.psse and engine.pypsa are then available if successfully initialized
```

If you only want PyPSA:
```python
engine.load(["pypsa"])
```

---

## Attaching WEC farms

Use `apply_wec()` to create and register a `WECFarm` against the loaded modelers.

```python
engine.apply_wec(
    farm_name="FarmA",
    size=4,              # number of devices
    sim_id=2,            # WECSim run id saved in DB
    model="RM3",         # or a filesystem path to a WEC model directory
    bus_location=31,     # the bus where the WEC generator(s) connect
    connecting_bus=1     # tie to swing bus by default (can be changed)
)
```

Notes:
- `model` can be either a **built-in model name** (e.g. `"RM3"`) or a **path** to a local WEC‑SIM model directory.
- `sim_id` links the farm to a specific WECSim result stored in the database.

---

## Generating load curves

You can synthesize per‑bus load time‑series with a normalized **double‑peak** (morning/evening) profile:

```python
prof = engine.generate_load_curves(
    morning_peak_hour=8.0,
    evening_peak_hour=18.0,
    amplitude=0.30,        # +/- 30% swing
    min_multiplier=0.70,   # clamp
    amp_overrides={31: 0.45}   # per‑bus amplitude overrides
)
```

- The profile length is derived from `engine.time.snapshots`.
- If the simulation window is short (< 6 hours), a flat profile is used.
- Base loads are pulled from the active backend (PSS®E or PyPSA).

---

## Simulation control

```python
engine.simulate(
    sim_length=None,   # If WEC data is present, length is capped to available WEC data
    load_curve=True,   # Build & apply synthetic load curves
    plot=True          # Let modelers render their plots
)
```

**Behavior with WEC data**  
If at least one WEC farm is present, the simulation length is automatically **capped** to the available WEC time‑series. If `sim_length` is supplied, the engine uses `min(sim_length, available_len)`.

---

## Running WEC‑SIM (optional)

You can drive WEC‑SIM via the MATLAB Engine through `WECSimRunner` (attached at `engine.wec_sim`).

1) Tell the runner where WEC‑SIM is installed:
```python
engine.wec_sim.wec_sim_path = r"C:\Users\me\WEC-Sim"   # must exist
```

2) Run the model and write outputs into the DB:
```python
ok = engine.wec_sim(
    sim_id=3,
    model="RM3",                  # or a full path to a model folder
    sim_length_secs=12*3600,      # 12 hours
    tsample=300,                  # 5 min downsample
    wave_height=2.5,
    wave_period=8.0
)
```

3) Attach the farm using the `sim_id` you just computed:
```python
if ok:
    engine.apply_wec("FarmAfterSim", size=8, sim_id=3, model="RM3", bus_location=31, connecting_bus=1)
```

> The runner will start the MATLAB engine, add the WEC‑SIM tree to the path, cd into the model directory, run the appropriate simulation function, write results to the DB, and (optionally) plot wave and power signals.

---

## API Reference

The sections below are **auto‑generated** from your code using `mkdocstrings`.  
Keep your docstrings up to date and rebuild the docs to refresh these pages.

### Engine

::: wecgrid.engine.Engine
    handler: python
    options:
      show_root_heading: false
      show_source: false

### Utilities (selected)

::: wecgrid.util.resources.resolve_grid_case
    handler: python
    options:
      show_root_heading: false
      show_source: false

---

## Tips & troubleshooting

- **PSS®E availability**: ensure the PSS®E Python API is importable in the same interpreter if you plan to use it.
- **WECSim path**: you must set `engine.wec_sim.wec_sim_path` before calling the runner.
- **Custom RAW files**: pass a full path to `engine.case()` to use your own grids.
- **DB location**: the default SQLite file is managed via `WECGridDB`; see `database.md` for details (optional page).

---

## API Reference

![mkapi](wecgrid.engine.Engine)
