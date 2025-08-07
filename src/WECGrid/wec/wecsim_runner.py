"""
Simulation runner for a WEC farm using WEC-SIM via MATLAB engine.
"""

import os
import matlab.engine
from typing import Optional, Dict, Any
from wecgrid.database.wecgrid_db import WECGridDB
from wecgrid.util.wecgrid_pathmanager import WECGridPathManager


# Inside wecsim_runner.py (at the top)
from dataclasses import dataclass


class WECSimRunner:
    def __init__(self, database: WECGridDB, path_manager: WECGridPathManager):
        """
        Args:
            wec_model_path: Path to root folder of all WEC models
            wec_sim_path: Path to the WEC-SIM framework
            db: An instance of WECGridDB for executing DB operations
        """

        self.path_manager: WECGridPathManager = path_manager
        self.database: WECGridDB = database
        self.matlab_engine: Optional[matlab.engine.MatlabEngine] = None


    def start_matlab(self) -> bool:
        """
        Starts a MATLAB Engine and adds WEC-SIM paths.
        """
        if self.matlab_engine is None:
            print("Starting MATLAB engine...")
            self.matlab_engine = matlab.engine.start_matlab()

            # Get and validate WEC-SIM path
            wec_sim_path = self.path_manager.wec_sim
            if wec_sim_path is None:
                raise ValueError("WEC-SIM path is not configured.")
            
            if not os.path.exists(wec_sim_path):
                raise FileNotFoundError(f"WEC-SIM path does not exist: {wec_sim_path}")

            matlab_path = self.matlab_engine.genpath(str(wec_sim_path), nargout=1)
            self.matlab_engine.addpath(matlab_path, nargout=0)
            print("MATLAB engine started and WEC-SIM path added...")
            return True
        else:
            print("MATLAB engine is already running.")
            return False
                
    def stop_matlab(self) -> bool:
        if self.matlab_engine is not None:
            self.matlab_engine.quit()
            self.matlab_engine = None
            print("MATLAB engine stopped.")
            return True
        print("MATLAB engine is not running.")
        return False
            
    def __call__(
        self,
        wec_id: int,
        model: str,
        sim_length: int = 300,
        tsample: float = 0.1,
        wave_height: float = 2.5,
        wave_period: float = 8.0,
        wave_seed: int = 42,
    ) -> bool:
        """
        Run WEC-SIM via MATLAB. Outputs are written to the database.

        Args:
            wec_id: Integer ID for the simulated device.
            model: Name of the WEC model (e.g. "RM3", "LUPA").
            sim_length: Simulation length in seconds.
            tsample: Time sample step in seconds.
            wave_height: Wave height in meters.
            wave_period: Wave period in seconds.
            wave_seed: Random seed for wave generation.

        Returns:
            True if simulation succeeds, False if error occurs.
        """
        try:
            if self.start_matlab():
                table_name = f"WEC_output_{wec_id}"
                with self.database.connection() as conn:
                    conn.cursor().execute(f"DROP TABLE IF EXISTS {table_name};")

                model_dir = os.path.join(self.path_manager.wec_models, model)
                self.matlab_engine.cd(model_dir)

                # Set simulation parameters in MATLAB workspace
                self.matlab_engine.workspace["wecId"] = wec_id
                self.matlab_engine.workspace["simLength"] = sim_length
                self.matlab_engine.workspace["Tsample"] = tsample
                self.matlab_engine.workspace["waveHeight"] = wave_height
                self.matlab_engine.workspace["wavePeriod"] = wave_period
                self.matlab_engine.workspace["waveSeed"] = wave_seed

                self.matlab_engine.workspace["DB_PATH"] = self.database.db_path

                # Run the appropriate WEC-SIM function
                if model.lower() == "lupa":
                    self.matlab_engine.eval(
                        "m2g_out = w2gSim_LUPA(wecId,simLength,Tsample,waveHeight,wavePeriod);",
                        nargout=0
                    )
                else:
                    self.matlab_engine.eval(
                        "m2g_out = w2gSim(wecId,simLength,Tsample,waveHeight,wavePeriod);",
                        nargout=0
                    )

                self.matlab_engine.eval("WECsim_to_PSSe_dataFormatter", nargout=0)
                print(f"WEC-SIM complete: model={model}, ID={wec_id}")
                #todo using the WECGridDB instance, we should double check if the data was written to the database
                #todo should add a data print or plot here to show the sim results
                self.stop_matlab()
                return True

            print("Failed to start MATLAB engine.")
            return False

        except Exception as e:
            print(f"[WEC-SIM ERROR] model={model}, ID={wec_id} → {e}")
            return False
