"""
Groups multiple WEC devices into a farm and manages their collective settings.
"""

from typing import List
from .device import Device   # <<<<< import Device here

class Farm:
    def __init__(self, devices: List[Device], layout: dict):
        """
        Args:
            devices: list of Device instances
            layout:  dict mapping device IDs to bus/location info
        """
        self.devices = devices
        self.layout  = layout
