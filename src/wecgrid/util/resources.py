# Python 3.7-safe import
try:
    from importlib.resources import files, as_file
except ImportError:  # 3.7 uses backport
    from importlib_resources import files, as_file

from pathlib import Path

def iter_grid_models():
    """Yield (name, resource) for bundled RAW cases."""
    pkg = "wecgrid.data.grid_models"
    base = files(pkg)
    # RAW files directly under grid_models/
    for res in base.iterdir():
        if res.name.lower().endswith(".raw"):
            yield res.name, res

def iter_wec_models():
    """Yield (name, directory resource) for bundled WEC-Sim models."""
    pkg = "wecgrid.data.wec_models"
    base = files(pkg)
    for res in base.iterdir():
        if res.is_dir():
            yield res.name, res

def resolve_grid_case(identifier: str) -> Path:
    """
    If identifier is a filesystem path to a .RAW, return it.
    Else, if it matches a bundled RAW filename (case-insensitive), return a temp-path to it.
    """
    p = Path(identifier)
    if p.exists() and p.suffix.lower() == ".raw":
        return p

    # try bundled
    wanted = identifier.lower()
    for name, res in iter_grid_models():
        if name.lower() == wanted or Path(name).stem.lower() == wanted:
            # as_file gives you a real filesystem path even from inside a zip
            with as_file(res) as fpath:
                return Path(fpath)

    raise FileNotFoundError(f"Grid case not found: {identifier}")

def resolve_wec_model(identifier: str) -> Path:
    """
    If identifier is a filesystem dir, return it.
    Else, if it matches a bundled WEC model dir, return a temp-path to that dir.
    """
    p = Path(identifier)
    if p.exists() and p.is_dir():
        return p

    wanted = identifier.lower()
    for name, res in iter_wec_models():
        if name.lower() == wanted:
            with as_file(res) as dpath:
                return Path(dpath)

    raise FileNotFoundError(f"WEC model not found: {identifier}")