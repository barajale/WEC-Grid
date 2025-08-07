# src/wecgrid/util/wecgrid_pathmanager.py

import json
import os
from pathlib import Path

class WECGridPathManager:
    def __init__(self, config_file: Path = None):
        if config_file is None:
            config_file = Path(__file__).parent / "paths.json"
        self.config_file = config_file
        self.project_root = Path(__file__).resolve().parents[2]
        self.paths = self._load_paths()
        self.check_all_paths()

    def _load_paths(self) -> dict:
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                return json.load(f)
        print(f"[ERROR] Config file not found: {self.config_file}")
        return {}

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else self.project_root / path

    def get_path(self, key: str) -> str:
        raw = self.paths.get(key)
        if not raw:
            raise ValueError(f"Path for '{key}' not set.")
        return str(self._resolve_path(raw))

    def set_path(self, key: str, path: str) -> None:
        self.paths[key] = path
        self._save_paths()

    def _save_paths(self) -> None:
        os.makedirs(self.config_file.parent, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self.paths, f, indent=2)

    def check_all_paths(self) -> None:
        for key, raw_path in self.paths.items():
            resolved = self._resolve_path(raw_path)
            if not resolved.exists():
                print(f"[WARNING] Path for '{key}' is missing or invalid: {resolved}")

    def __getattr__(self, name: str) -> str:
        # Enable dot-access to path keys
        if name in self.paths:
            return self.get_path(name)
        raise AttributeError(f"'WECGridPathManager' has no attribute '{name}'")