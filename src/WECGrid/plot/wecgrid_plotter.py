"""
WEC-Grid high-level plotting interface
"""

class WECGridPlotter:
    def __init__(self, engine):
        """
        Args:
            engine: the WEC-Grid Engine instance
        """
        self.engine = engine

    # add your top-level plot methods here, e.g.:
    # def plot(self, component, *args, **kwargs): ...