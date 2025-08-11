"""
Simulation runner for a WEC farm using WEC-SIM via MATLAB engine.
"""

import os
import random

import matlab.engine
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from wecgrid.database.wecgrid_db import WECGridDB
#from wecgrid.util.wecgrid_pathmanager import WECGridPathManager
from wecgrid.util.resources import resolve_wec_model


# Inside wecsim_runner.py (at the top)
from dataclasses import dataclass


class WECSimRunner:
    def __init__(self, database: WECGridDB):
        """
        Args:
            wec_model_path: Path to root folder of all WEC models
            wec_sim_path: Path to the WEC-SIM framework
            db: An instance of WECGridDB for executing DB operations
        """

        #self.path_manager: WECGridPathManager = path_manager
        self.wec_sim_path: Optional[str] = None
        self.database: WECGridDB = database
        self.matlab_engine: Optional[matlab.engine.MatlabEngine] = None
    
    def set_wec_sim_path(self, path: str) -> None:
        self.wec_sim_path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"WEC-SIM path does not exist: {path}")
        


    def start_matlab(self) -> bool:
        """
        Starts a MATLAB Engine and adds WEC-SIM paths.
        """
        if self.matlab_engine is None:
            print("Starting MATLAB engine...")
            self.matlab_engine = matlab.engine.start_matlab()

            # Get and validate WEC-SIM path
            if self.wec_sim_path is None:
                raise ValueError("WEC-SIM path is not configured. Please set it using set_wec_sim_path()")
            wec_sim_path = self.wec_sim_path
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

    def sim_results(self, df_full, df_ds, model, sim_id):
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Secondary y-axis: Wave Height (m) — drawn first for background
        ax2 = ax1.twinx()
        ax2.set_ylabel("Wave Height (m)")
        ax2.plot(
            df_full["time"], df_full["eta"],
            color="tab:blue", alpha=0.3, linewidth=1, label="Wave Height"
        )

        # Primary y-axis: Active power (MW)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Active Power (MW)")
        ax1.plot(df_full["time"], df_full["p"], color="gray", label="P (full)", linewidth=1)
        ax1.plot(
            df_ds["time"], df_ds["p"],
            linestyle=":", marker="o", color="tab:red", label="P (downsampled)"
        )

        # Title + layout
        fig.suptitle(f"WEC-SIM Output — Model: {model}, Sim ID: {sim_id}")
        fig.tight_layout()

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

        plt.show()
            
    def __call__(
        self,
        sim_id: int,
        model: str,
        sim_length_secs: int = 3600 * 24, # 24 hours
        tsample: float = 300,
        wave_height: float = 2.5,
        wave_period: float = 8.0,
        wave_seed: int = random.randint(1, 100),
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
        #TODO some sorta sim progress bar would be cool? 
        
        try:
            model_dir = resolve_wec_model(model)  # accepts name or path
            
            if self.start_matlab():
                table_name = f"WECSIM_{model.lower()}_{sim_id}"
                with self.database.connection() as conn:
                    conn.cursor().execute(f"DROP TABLE IF EXISTS {table_name};")

                print("Starting WEC-SIM simulation...")
                #model_dir = os.path.join(self.path_manager.wec_models, model)
                self.matlab_engine.cd(str(model_dir))

                # Set simulation parameters in MATLAB workspace
                self.matlab_engine.workspace["sim_id"] = sim_id
                self.matlab_engine.workspace["model"] = model.lower()
                self.matlab_engine.workspace["simLength"] = sim_length_secs
                self.matlab_engine.workspace["Tsample"] = tsample
                self.matlab_engine.workspace["waveHeight"] = wave_height
                self.matlab_engine.workspace["wavePeriod"] = wave_period
                self.matlab_engine.workspace["waveSeed"] = wave_seed

                self.matlab_engine.workspace["DB_PATH"] = self.database.db_path

                # Run the appropriate WEC-SIM function
                if model.lower() == "lupa":
                    self.matlab_engine.eval(
                        "m2g_out = w2gSim_LUPA(sim_id,simLength,Tsample,waveHeight,wavePeriod, model);",
                        nargout=0
                    )
                else:
                    self.matlab_engine.eval(
                        "m2g_out = w2gSim(sim_id,simLength,Tsample,waveHeight,wavePeriod,waveSeed,model);",
                        nargout=0
                    )
                print("simulation complete... writing to database")

                self.matlab_engine.eval("WECsim_to_PSSe_dataFormatter", nargout=0)
                print(f"WEC-SIM complete: model = {model}, ID = {sim_id}, duration = {sim_length_secs}s")
                #todo using the WECGridDB instance, we should double check if the data was written to the database
                #todo should add a data print or plot here to show the sim results
                self.stop_matlab()
                
                df_ds = self.database.query(f"SELECT * FROM {table_name}", return_type="df")
                df_full = self.database.query(f"SELECT * FROM {table_name}_full", return_type="df")

                self.sim_results(df_full, df_ds, model, sim_id)
                return True

            print("Failed to start MATLAB engine.")
            return False

        except Exception as e:
            print(f"[WEC-SIM ERROR] model={model}, ID={sim_id} → {e}")
            return False
