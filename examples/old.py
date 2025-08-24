import sys, os

# Add PSSE paths
os.environ["PATH"] = r"C:\Program Files (x86)\PTI\PSSE34\PSSBIN;" + os.environ["PATH"]
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE34\PSSPY34")

# Add your package path (src directory)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import psspy
import psse34

# Now you can import your wecgrid package
import wecgrid

engine = wecgrid.Engine()

engine.case("./grid/RTS-GMLC_Hooman.raw")
engine.load(["psse"])
engine.database.set_database_path("./new_WECGrid.db")
engine.apply_wec(
   farm_name = "WEC-Farm",
   size = 10, # one RM3 in WEC farm  
   wec_sim_id = 1, # RM3 run id  
   bus_location=326, # create a new bus for farm  
   connecting_bus = 123, # Connect to bus 1 or swing bus
   scaling_factor = 1 # scale up the lab scale to about a 1kW
)

engine.simulate(
    load_curve=False
)

engine.database.save_sim(sim_name="PSSE-RTS-GMLC: RM3 Farm", 
                         notes="RTS-GMLC grid simulation using PSS/E. The simulation was run for 24 hours at 5-minute resolution with no load curve applied. A WEC Farm with 10 RM3 WEC models was included in this simulation using the RM3 WEC-Sim run id = 1."
)