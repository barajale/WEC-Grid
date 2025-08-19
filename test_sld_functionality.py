#!/usr/bin/env python3
"""
Test script for the new GridState-based Single Line Diagram (SLD) functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from wecgrid.plot.plot import WECGridPlot

def create_mock_engine_with_gridstate():
    """Create a mock engine with GridState data for testing SLD."""
    
    class MockGrid:
        def __init__(self):
            # Create mock bus data - small 4-bus system
            self.bus = pd.DataFrame({
                'bus': [1, 2, 3, 4],
                'bus_name': ['Bus_1', 'Bus_2', 'Bus_3', 'Bus_4'],
                'type': ['Slack', 'PV', 'PQ', 'PQ'],
                'p': [0.5, -0.3, 0.0, 0.0],  # pu
                'q': [0.2, -0.1, 0.0, 0.0],  # pu
                'v_mag': [1.0, 1.05, 0.98, 0.96],  # pu
                'angle_deg': [0.0, -2.5, -5.2, -8.1],
                'Vbase': [138.0, 138.0, 69.0, 69.0]  # kV
            })
            
            # Create mock line data - connecting buses
            self.line = pd.DataFrame({
                'line': [1, 2, 3],
                'line_name': ['Line_1', 'Line_2', 'Line_3'],
                'ibus': [1, 1, 2],
                'jbus': [2, 3, 4],
                'line_pct': [45.2, 32.1, 28.7],  # % loading
                'status': [1, 1, 1]  # all active
            })
            
            # Create mock generator data
            self.gen = pd.DataFrame({
                'gen': [1, 2],
                'gen_name': ['Gen_1', 'Gen_2'],
                'bus': [1, 2],
                'p': [0.8, 0.6],  # pu
                'q': [0.3, 0.2],  # pu
                'Mbase': [100.0, 75.0],  # MVA
                'status': [1, 1]  # both online
            })
            
            # Create mock load data
            self.load = pd.DataFrame({
                'load': [1, 2],
                'load_name': ['Load_1', 'Load_2'],
                'bus': [3, 4],
                'p': [0.4, 0.3],  # pu
                'q': [0.15, 0.12],  # pu
                'status': [1, 1]  # both connected
            })
    
    class MockModeler:
        def __init__(self):
            self.grid = MockGrid()
    
    class MockEngine:
        def __init__(self):
            self.case_name = "Test_4_Bus_System"
            self.psse = MockModeler()
            self.pypsa = MockModeler()
    
    return MockEngine()

def test_sld_functionality():
    """Test the SLD generation with mock data."""
    print("=== Testing GridState-based SLD Generation ===\n")
    
    # Create mock engine
    engine = create_mock_engine_with_gridstate()
    
    # Create plotter
    plotter = WECGridPlot(engine)
    
    print("Mock system created:")
    print(f"  Buses: {len(engine.psse.grid.bus)}")
    print(f"  Lines: {len(engine.psse.grid.line)}")
    print(f"  Generators: {len(engine.psse.grid.gen)}")
    print(f"  Loads: {len(engine.psse.grid.load)}")
    
    # Test SLD generation for PSS®E backend
    print(f"\nGenerating SLD for PSS®E backend...")
    try:
        fig_psse = plotter.sld("psse", title="Test SLD - PSS®E Backend")
        print("✅ PSS®E SLD generated successfully!")
    except Exception as e:
        print(f"❌ PSS®E SLD failed: {e}")
        return False
    
    # Test SLD generation for PyPSA backend  
    print(f"\nGenerating SLD for PyPSA backend...")
    try:
        fig_pypsa = plotter.sld("pypsa", title="Test SLD - PyPSA Backend")
        print("✅ PyPSA SLD generated successfully!")
    except Exception as e:
        print(f"❌ PyPSA SLD failed: {e}")
        return False
    
    print(f"\n🎉 SLD functionality test completed successfully!")
    
    print(f"\n📝 SLD Features Available:")
    print(f"  ✓ Bus representation with type-based coloring")
    print(f"  ✓ Transmission line connections")
    print(f"  ✓ Generator symbols (circles with 'G')")
    print(f"  ✓ Load symbols (downward arrows)")
    print(f"  ✓ Automatic network layout using NetworkX")
    print(f"  ✓ Legend with component identification")
    print(f"  ✓ Works with both PSS®E and PyPSA data")
    
    print(f"\n⚠️  Current Limitations:")
    print(f"  • No transformer identification (data not in GridState)")
    print(f"  • Algorithmic layout (not geographical)")
    print(f"  • No shunt devices (not in current schema)")
    print(f"  • No line impedance/rating information display")
    
    return True

if __name__ == "__main__":
    try:
        success = test_sld_functionality()
        if success:
            print(f"\n🚀 Ready to use:")
            print(f"   from wecgrid.plot.plot import WECGridPlot")
            print(f"   plotter = WECGridPlot(engine)")
            print(f"   plotter.sld('psse')  # or 'pypsa'")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
