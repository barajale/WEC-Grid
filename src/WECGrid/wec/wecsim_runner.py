"""
Simulation runner for a WEC farm using WEC-SIM via MATLAB engine.
"""

import os
import matlab.engine
from typing import Dict, Any
from wecgrid.database.wecgrid_db import dbQuery, DB_PATH


class WECSimRunner:
    def __init__(self, wec_model_path: str, wec_sim_path: str):
        """
        Args:
            wec_model_path: Path to root folder of all WEC models
            wec_sim_path: Path to the WEC-SIM framework
        """
        self.wec_model_path = wec_model_path
        self.wec_sim_path = wec_sim_path
        self.eng = None 

    def start_matlab(self):
        if self.eng is None:
            print("Starting MATLAB engine...")
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(self.eng.genpath(self.wec_sim_path), nargout=0)
            print("MATLAB engine ready.")

    def __call__(self, sim_id: int, model: str, config: Dict[str, Any]) -> bool:
        """
        Run WEC-SIM via MATLAB. Outputs are written to the database.

        Args:
            sim_id: Integer ID for the simulated device.
            model: Folder name of the WEC model (e.g. "RM3").
            config: Simulation config dict (e.g., simLength, Tsample, waveHeight, wavePeriod).

        Returns:
            True if simulation succeeds, False if error occurs.
        """
        try:
            self.start_matlab()

            # Drop any existing data for this sim_id
            table_name = f"WEC_output_{sim_id}"
            dbQuery(f"DROP TABLE IF EXISTS {table_name};")

            # Set up MATLAB workspace
            self.eng.cd(os.path.join(self.wec_model_path, model))
            self.eng.workspace["wecId"] = sim_id
            for key, value in config.items():
                self.eng.workspace[key] = value
            self.eng.workspace["DB_PATH"] = DB_PATH

            # Run simulation
            if model.lower() == "lupa":
                self.eng.eval("m2g_out = w2gSim_LUPA(wecId,simLength,Tsample,waveHeight,wavePeriod);", nargout=0)
            else:
                self.eng.eval("m2g_out = w2gSim(wecId,simLength,Tsample,waveHeight,wavePeriod);", nargout=0)

            # Format + write to DB
            self.eng.eval("WECsim_to_PSSe_dataFormatter", nargout=0)
            print(f"WEC-SIM complete: model={model}, ID={sim_id}")
            return True

        except Exception as e:
            print(f"[WEC-SIM ERROR] model={model}, ID={sim_id} → {e}")
            return False