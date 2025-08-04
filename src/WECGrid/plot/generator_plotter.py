"""
Generator-level plotting routines for WEC-Grid
"""

class GeneratorPlotter:
    def __init__(self, engine):
        """
        Args:
            engine: the WEC-Grid Engine instance
        """
        self.engine = engine

    # e.g. def plot_generator_power(self, gen_id): ...