"""
Bus-level plotting routines for WEC-Grid
"""

class BusPlotter:
    def __init__(self, engine):
        """
        Args:
            engine: the WEC-Grid Engine instance
        """
        self.engine = engine

    # e.g. def plot_bus_power(self, bus_number): ...
    #      def plot_bus_voltage(self, bus_number): ...