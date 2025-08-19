"""
WEC-Grid high-level plotting interface

This module provides comprehensive visualization capabilities for WEC-GRID simulation
results, supporting cross-platform comparison between PSS®E and PyPSA modeling backends.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import networkx as nx
from typing import Any, List, Union, Optional


class WECGridPlot:
    """High-level plotting interface for WEC-GRID simulation visualization.
    
    Provides comprehensive visualization tools for analyzing power system simulation
    results from PSS®E and PyPSA backends. Supports time-series plotting, cross-platform
    comparison, WEC farm analysis, and statistical validation between modeling tools.
    
    Args:
        engine: WEC-GRID Engine instance with simulation results from backends.
    
    Attributes:
        engine: Reference to WEC-GRID Engine with loaded simulation data.
        
    Example:
        >>> plotter = WECGridPlotter(engine)
        >>> plotter.plot_generator("psse", "p")  # Generator power
        >>> plotter.compare_software("bus", "v_mag")  # Cross-platform comparison
        >>> plotter.plot_wec_analysis()  # WEC farm analysis
        
    Notes:
        - Supports both PSS®E and PyPSA simulation backends
        - Provides statistical comparison metrics (MSE, RMSE, MAE, MAPE, R²)
        - Generates publication-quality figures with customizable styling
        - Includes specialized WEC farm analysis capabilities
    """
    
    def __init__(self, engine: Any):
        """Initialize WECGridPlotter with WEC-GRID Engine.
        
        Args:
            engine: WEC-GRID Engine with 'psse' and/or 'pypsa' attributes.
        """
        self.engine = engine
    
    def plot_component_data(self, software, component_type, parameter, component_name=None, 
                           bus=None, figsize=(12, 6), style='default', show_grid=True, 
                           save_path=None, **plot_kwargs):
        """Plot time series data for grid components.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            component_type (str): Component type ("gen", "bus", "line", "load").
            parameter (str): Parameter to plot ("p", "q", "v_mag", "angle_deg", "line_pct").
            component_name (str or list, optional): Specific component(s) to plot.
            bus (int or list, optional): Bus number(s) to filter components by location.
            figsize (tuple, optional): Figure size (width, height). Defaults to (12, 6).
            style (str, optional): Matplotlib style name. Defaults to 'default'.
            show_grid (bool, optional): Whether to show grid lines. Defaults to True.
            save_path (str, optional): Path to save figure. Defaults to None.
            **plot_kwargs: Additional matplotlib plotting arguments.
        
        Returns:
            tuple: (fig, ax) matplotlib objects, or (None, None) if plotting fails.
                
        Raises:
            ValueError: If software, component_type, or parameter invalid.
            AttributeError: If engine missing specified backend or grid data.
        
        Example:
            >>> # Plot all generator active power
            >>> fig, ax = plotter.plot_component_data("psse", "gen", "p")
            >>> 
            >>> # Plot generators at specific buses
            >>> plotter.plot_component_data("pypsa", "gen", "p", bus=[1, 2, 14])
            >>> 
            >>> # Plot specific generator by name
            >>> plotter.plot_component_data("pypsa", "gen", "p", component_name=["Gen_1"])
            
        Notes:
            - Automatically handles data validation and missing components
            - Y-axis labels set based on parameter type
            - Legend managed intelligently for multiple components
            - Supports filtering by bus location or component name
        """
        
        # Set style with error handling
        try:
            plt.style.use(style)
        except OSError:
            print(f"Style '{style}' not found. Using default style.")
            print(f"Available styles: {plt.style.available}")
            plt.style.use('default')
        
        # Validate inputs
        if software not in ["psse", "pypsa"]:
            raise ValueError("Software must be 'psse' or 'pypsa'")
        
        # Get the appropriate grid object
        try:
            grid_obj = getattr(self.engine, software).grid
        except AttributeError:
            raise ValueError(f"Engine does not have {software} attribute or grid is not loaded")
        
        # Get component data
        component_attr = f"{component_type}_t"
        if not hasattr(grid_obj, component_attr):
            raise ValueError(f"Grid object does not have {component_attr} attribute")
        
        component_data = getattr(grid_obj, component_attr)
        
        # Get parameter data
        if not hasattr(component_data, parameter):
            available_params = [attr for attr in dir(component_data) if not attr.startswith('_')]
            raise ValueError(f"Parameter '{parameter}' not found. Available: {available_params}")
        
        data = getattr(component_data, parameter)
        

        # Resolve component selection based on names and/or bus filtering
        selected_components = self._resolve_component_selection(
            grid_obj, component_type, component_name, bus
        )
        
        # Handle component selection
        if selected_components is not None:
            # Filter data to only include selected components
            available_cols = list(data.columns)
            
            # Direct name matching since time-series now uses component names
            valid_components = [comp for comp in selected_components if comp in available_cols]
            
            if not valid_components:
                print(f"Error: No valid components found to plot.")
                print(f"Requested components: {selected_components}")
                print(f"Available columns: {available_cols}")
                return None, None
            
            if len(valid_components) != len(selected_components):
                missing = [comp for comp in selected_components if comp not in valid_components]
                print(f"Warning: Some components not found: {missing}")
                print(f"Found: {valid_components}")
            
            data = data[valid_components]
        #     #print(f"Plotting {len(valid_components)} selected components: {valid_components}")
        # else:
        #     #print(f"Plotting all {len(data.columns)} components")
        
        # Add data summary
        if isinstance(data, pd.DataFrame):
            #print(f"Plotting {len(data.columns)} components over {len(data)} time steps")
            if data.empty:
                print("Warning: No data to plot")
                return None, None
        # else:
        #     print(f"Plotting single component over {len(data)} time steps")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set default plot kwargs
        default_kwargs = {'linewidth': 2, 'alpha': 0.8}
        default_kwargs.update(plot_kwargs)
        
        # Plot the data
        if isinstance(data, pd.Series):
            ax.plot(data.index, data.values, label=data.name, **default_kwargs)
        else:
            data.plot(ax=ax, **default_kwargs)
        
        # Customize the plot
        title = f"{component_type.upper()} {parameter.upper()} - {software.upper()}"
        if selected_components is not None:
            if bus is not None:
                bus_str = f"Bus {bus}" if isinstance(bus, int) else f"Buses {bus}"
                title += f" ({bus_str})"
            if component_name is not None:
                comp_str = ', '.join(map(str, component_name)) if len(component_name) <= 3 else f'{len(component_name)} components'
                title += f" - {comp_str}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        
        # Set y-label based on parameter
        ylabel_map = {
            'p': 'Active Power (MW)',
            'q': 'Reactive Power (MVAr)', 
            'v': 'Voltage (p.u.)',
            'v_mag': 'Voltage Magnitude (p.u.)',
            'angle_deg': 'Voltage Angle (degrees)',
            'line_pct': 'Line Loading (%)',
            'pf': 'Power Factor',
            'freq': 'Frequency (Hz)'
        }
        ax.set_ylabel(ylabel_map.get(parameter, parameter.upper()), fontsize=12)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Add legend if multiple components
        if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        #plt.show()
        
        return fig, ax

    def _filter_components_by_bus(self, grid_obj, component_type: str, bus_list: List[int]) -> List[str]:
        """Filter components by their bus location.
        
        Args:
            grid_obj: Grid object with snapshot data
            component_type (str): Component type ("gen", "load", "line")
            bus_list (List[int]): List of bus numbers to filter by
            
        Returns:
            List[str]: List of component names connected to specified buses
        """
        snapshot_attr = getattr(grid_obj, component_type, None)
        if snapshot_attr is None or snapshot_attr.empty:
            print(f"Warning: No snapshot data found for {component_type}")
            return []
        
        # Different components have different bus connection columns
        if component_type == "gen":
            bus_col = "bus"
        elif component_type == "load":
            bus_col = "bus" 
        elif component_type == "line":
            # Lines connect two buses - include if either ibus or jbus matches
            if "ibus" in snapshot_attr.columns and "jbus" in snapshot_attr.columns:
                mask = (snapshot_attr["ibus"].isin(bus_list)) | (snapshot_attr["jbus"].isin(bus_list))
                # Return component names instead of IDs
                name_col = f"{component_type}_name"
                if name_col in snapshot_attr.columns:
                    filtered_names = snapshot_attr.loc[mask, name_col].tolist()
                else:
                    filtered_names = snapshot_attr.index[mask].astype(str).tolist()
                print(f"Found {len(filtered_names)} {component_type} components at buses {bus_list}: {filtered_names}")
                return filtered_names
            else:
                print(f"Warning: Line data missing ibus/jbus columns")
                return []
        else:
            print(f"Warning: Unknown component type: {component_type}")
            return []
        
        if bus_col in snapshot_attr.columns:
            mask = snapshot_attr[bus_col].isin(bus_list)
            # Return component names instead of IDs
            name_col = f"{component_type}_name"
            if name_col in snapshot_attr.columns:
                filtered_names = snapshot_attr.loc[mask, name_col].tolist()
            else:
                # Fallback to IDs if no name column
                filtered_names = snapshot_attr.index[mask].astype(str).tolist()
            print(f"Found {len(filtered_names)} {component_type} components at buses {bus_list}: {filtered_names}")
            return filtered_names
        else:
            print(f"Warning: {component_type} data missing {bus_col} column")
            return []

    def _resolve_component_selection(self, grid_obj, component_type: str, 
                                   component_name: Optional[Union[str, List[str]]] = None,
                                   bus: Optional[Union[int, List[int]]] = None) -> Optional[List[str]]:
        """Resolve component selection based on names and/or bus locations.
        
        Args:
            grid_obj: Grid object with snapshot data
            component_type (str): Component type ("gen", "load", "line", "bus")
            component_name (Optional[Union[str, List[str]]]): Specific component names
            bus (Optional[Union[int, List[int]]]): Bus numbers to filter by
            
        Returns:
            Optional[List[str]]: List of component names to plot, or None for all components
        """
        selected_components = []
        
        # Handle bus filtering
        if bus is not None:
            if isinstance(bus, int):
                bus = [bus]
            
            print(f"Filtering {component_type} by buses: {bus}")
            
            if component_type == "bus":
                # For buses, the bus numbers are the component names
                selected_components.extend([str(b) for b in bus])
            else:
                # For other components, find component names connected to specified buses
                bus_filtered = self._filter_components_by_bus(grid_obj, component_type, bus)
                selected_components.extend(bus_filtered)
        
        # Handle component name filtering
        if component_name is not None:
            if isinstance(component_name, str):
                component_name = [component_name]
            
            print(f"Filtering {component_type} by names: {component_name}")
            
            if selected_components:
                # If we already have bus-filtered components, take intersection
                intersection = [comp for comp in selected_components if comp in component_name]
                selected_components = intersection
                print(f"Intersection of bus and name filters: {selected_components}")
            else:
                # Otherwise, use name filtering
                selected_components = component_name
        
        # Return None if no specific filtering requested (plot all)
        if not selected_components and bus is None and component_name is None:
            return None
            
        print(f"Final component selection for {component_type}: {selected_components}")
        return selected_components if selected_components else []

    def gen(self, software='pypsa', parameter='p', bus=None, component_name=None, **kwargs):
        """Plot generator time-series data.
        
        Convenience method for plotting generator parameters using the main
        plot_component_data method with component_type='gen'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Generator parameter to plot. Defaults to 'p'.
                Common values: 'p' (active power), 'q' (reactive power), 'status'.
            bus (int or list, optional): Bus number(s) to filter generators by location.
            component_name (str or list, optional): Specific generator(s) to plot by name.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all generator active power
            >>> plotter.generator("psse", "p")
            >>> # Plot generators at specific buses
            >>> plotter.generator("pypsa", "p", bus=[1, 2, 31])
            >>> # Plot specific generators by name
            >>> plotter.generator("pypsa", "q", component_name=["Gen_1", "Gen_2"])
        """
        return self.plot_component_data(software, 'gen', parameter, component_name=component_name, bus=bus, **kwargs)

    def bus(self, software='pypsa', parameter='p', bus=None, **kwargs):
        """Plot bus time-series data.
        
        Convenience method for plotting bus parameters using the main
        plot_component_data method with component_type='bus'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Bus parameter to plot. Defaults to 'p'.
                Common values: 'p' (net power), 'q' (net reactive power), 
                'v_mag' (voltage magnitude), 'angle_deg' (voltage angle).
            bus (list of int, optional): Bus number to filter generators by location.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all bus voltage magnitudes
            >>> plotter.plot_bus("psse", "v_mag")
            >>> # Plot specific bus power injections
            >>> plotter.plot_bus("pypsa", "p", component_name=[1, 2, 14])
        """
        return self.plot_component_data(software, 'bus', parameter, component_name=None, bus=bus, **kwargs)

    def line(self, software='pypsa', parameter='line_pct', bus=None, component_name=None,  **kwargs):
        """Plot transmission line time-series data.
        
        Convenience method for plotting line parameters using the main
        plot_component_data method with component_type='line'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Line parameter to plot. Defaults to 'line_pct'.
                Common values: 'line_pct' (loading percentage), 'status'.
            component_name (list of str, optional): Specific line(s) to plot.
            bus (list of int, optional): Bus number to filter generators by location.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all line loading percentages
            >>> plotter.plot_line("psse", "line_pct")
            >>> # Plot specific line status
            >>> plotter.plot_line("pypsa", "status", component_name=["Line_1_2_1"])
        """
        return self.plot_component_data(software, 'line', parameter, component_name=component_name, bus=bus, **kwargs)

    def load(self, software='pypsa', parameter='p', bus = None, component_name=None, **kwargs):
        """Plot load time-series data.
        
        Convenience method for plotting load parameters using the main
        plot_component_data method with component_type='load'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Load parameter to plot. Defaults to 'p'.
                Common values: 'p' (active power), 'q' (reactive power), 'status'.
            component_name (str or list, optional): Specific load(s) to plot.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all load active power consumption
            >>> plotter.plot_load("psse", "p")
            >>> # Plot specific load reactive power
            >>> plotter.plot_load("pypsa", "q", component_name=["Load_1_1", "Load_2_1"])
        """
        return self.plot_component_data(software, 'load', parameter, component_name=component_name, bus=bus, **kwargs)

    def plot_component_grid(self, software, component_type, parameters, 
                           component_name=None, figsize=(15, 10), **kwargs):
        """Plot multiple parameters in a grid layout for comprehensive analysis.
        
        Creates a multi-subplot figure showing different parameters for the same
        component type, enabling quick visual comparison of multiple electrical
        quantities simultaneously.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            component_type (str): Component type ("gen", "bus", "line", "load").
            parameters (list): List of parameter names to plot.
            component_name (str or list, optional): Specific component(s) to plot.
                If None, plots all available components.
            figsize (tuple, optional): Figure size (width, height). Defaults to (15, 10).
            **kwargs: Additional matplotlib plotting arguments.
        
        Returns:
            tuple: (fig, axes) matplotlib figure and axes array.
            
        Example:
            >>> # Plot multiple bus parameters
            >>> parameters = ['v_mag', 'angle_deg', 'p', 'q']
            >>> plotter.plot_component_grid("psse", "bus", parameters)
            >>> 
            >>> # Plot generator parameters for specific units
            >>> gen_params = ['p', 'q', 'status']
            >>> plotter.plot_component_grid(
            ...     "pypsa", "gen", gen_params, 
            ...     component_name=["1_1", "2_1"]
            ... )
            
        Notes:
            - Automatically arranges subplots in a 2-column grid
            - Handles error conditions gracefully with informative messages
            - Applies consistent styling across all subplots
            - Hides unused subplot areas for clean presentation
        """
        n_params = len(parameters)
        cols = 2
        rows = (n_params + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        ylabel_map = {
            'p': 'Active Power (MW)',
            'q': 'Reactive Power (MVAr)', 
            'v': 'Voltage (p.u.)',
            'v_mag': 'Voltage Magnitude (p.u.)',
            'angle_deg': 'Voltage Angle (degrees)',
            'line_pct': 'Line Loading (%)',
            'pf': 'Power Factor',
            'freq': 'Frequency (Hz)'
        }
        
        for i, param in enumerate(parameters):
            ax = axes[i]
            try:
                grid_obj = getattr(self.engine, software).grid
                component_data = getattr(grid_obj, f"{component_type}_t")
                data = getattr(component_data, param)
                
                if component_name:
                    data = data[component_name] if component_name in data.columns else data
                
                data.plot(ax=ax, **kwargs)
                ax.set_title(f"{param.upper()}", fontweight='bold')
                ax.set_ylabel(ylabel_map.get(param, param.upper()))
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        
        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f"{component_type.upper()} Parameters - {software.upper()}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        #plt.show()
        return fig, axes


    def plot_wec_analysis(self, software="pypsa", figsize=(12, 15), show=True):
        """Generate comprehensive WEC farm analysis plots.
        
        Creates a specialized three-panel analysis plot focusing on Wave Energy Converter
        (WEC) farm performance and grid integration effects. This method provides
        publication-quality visualizations for WEC integration studies.
        
        Args:
            software (str, optional): Backend software to use. Defaults to "pypsa".
            figsize (tuple, optional): Figure size (width, height). Defaults to (12, 15).
            show (bool, optional): Whether to display the plot. Defaults to True.
        
        Returns:
            tuple: (fig, axes) matplotlib figure and axes array, or (None, None)
                if no WEC farms are found.
                
        Raises:
            AttributeError: If engine does not have WEC farms or grid data.
            
        Plot Panels:
            1. **WEC Farm Active Power Output**: Time series of power generation
               from all WEC farms [MW].
            2. **WEC Contribution Percentage**: WEC farms' contribution to total
               system generation over time [%].
            3. **WEC Farm Bus Voltage**: Voltage profiles at WEC connection
               points [p.u.].
        
        Example:
            >>> # Generate WEC analysis for PSS®E results
            >>> fig, axes = plotter.plot_wec_analysis("psse")
            >>> 
            >>> # Generate WEC analysis for PyPSA results with custom size
            >>> fig, axes = plotter.plot_wec_analysis("pypsa", figsize=(10, 12))
            
        Notes:
            - Uses GridState data to find generators at WEC farm bus locations
            - Identifies WEC generators by bus_location from farm data
            - Calculates meaningful metrics like contribution percentages
            - Handles missing data gracefully with informative error messages
            - Provides console output showing detected WEC farms
            - Designed for publication-quality WEC integration analysis
            
        WEC Farm Requirements:
            Each WEC farm object must have:
            - bus_location: Bus number where farm is connected
            - farm_name: Human-readable farm name for plots (if available)
        """
        if not hasattr(self.engine, 'wec_farms') or not self.engine.wec_farms:
            print("No WEC farms found in engine")
            return None, None
        
        try:
            # Get GridState data 
            grid_state = getattr(self.engine, software).grid
            gen_data = getattr(grid_state, 'gen_t')
            bus_data = getattr(grid_state, 'bus_t')
            
            if not hasattr(grid_state, 'time_series') or grid_state.time_series.empty:
                print("No time series data found in GridState")
                return None, None
 

            for farm in self.engine.wec_farm:
                fig, axes = plt.subplots(3, 1, figsize=figsize)
                
                p_data = getattr(gen_data, 'p')
                gen_data = getattr(p_data, ) # generator name 'G0'..'G6'
                
                # Plot WEC power data
                wec_df = pd.DataFrame(gen_data)
                wec_df.plot(ax=axes[0], linewidth=2)
                axes[0].set_title("WEC-Farm Active Power Output")
                axes[0].set_ylabel("Active Power (MW)")
                axes[0].grid(True, alpha=0.3)
                if len(gen_data) > 1:
                    axes[0].legend(loc='upper right')
                    
            # # Plot 2: WEC Contribution Percentage over Time
            # if wec_power_data:
            #     wec_df = pd.DataFrame(wec_power_data)
            #     total_wec_power = wec_df.sum(axis=1)
                
            #     # Calculate total generation from all gen_p_* columns
            #     all_gen_cols = [col for col in ts_data.columns if col.startswith('gen_p_')]
            #     if all_gen_cols:
            #         total_gen_power = ts_data[all_gen_cols].sum(axis=1)
            #         wec_percentage = (total_wec_power / total_gen_power) * 100
                    
            #         axes[1].plot(wec_percentage.index, wec_percentage.values, 
            #                     label="WEC Contribution %", linewidth=2, color='green')
            #         axes[1].set_title("WEC-Farm Contribution to Total Generation (%)")
            #         axes[1].set_ylabel("Percentage (%)")
            #         axes[1].legend()
            #         axes[1].grid(True, alpha=0.3)
            #     else:
            #         axes[1].text(0.5, 0.5, "No total generation\ndata for comparison", 
            #                     ha='center', va='center', transform=axes[1].transAxes)
            #         axes[1].set_title("WEC-Farm Contribution to Total Generation (%)")
            # else:
            #     axes[1].text(0.5, 0.5, "No WEC power data\nfor percentage calculation", 
            #                 ha='center', va='center', transform=axes[1].transAxes)
            #     axes[1].set_title("WEC-Farm Contribution to Total Generation (%)")
            
            # # Plot 3: WEC-Farm Bus Voltage
            # wec_buses = list(set(info['bus_location'] for info in wec_farm_info))
            # wec_voltage_data = {}
            
            # # Find bus voltage columns (look for bus_v_* patterns)
            # for col in ts_data.columns:
            #     if col.startswith('bus_v_'):
            #         bus_name = col.replace('bus_v_', '')
            #         # Try to match bus numbers
            #         try:
            #             bus_num = int(bus_name)
            #             if bus_num in wec_buses:
            #                 # Find corresponding farm name for this bus
            #                 farm_names = [info['farm_name'] for info in wec_farm_info 
            #                             if info['bus_location'] == bus_num]
            #                 if farm_names:
            #                     label = f"{farm_names[0]} (Bus {bus_num})"
            #                     wec_voltage_data[label] = ts_data[col]
            #         except ValueError:
            #             # Handle non-numeric bus names
            #             continue
            
            # if wec_voltage_data:
            #     voltage_df = pd.DataFrame(wec_voltage_data)
            #     voltage_df.plot(ax=axes[2], linewidth=2)
            #     axes[2].set_title("WEC-Farm Bus Voltage")
            #     axes[2].set_ylabel("Voltage (p.u.)")
            #     axes[2].grid(True, alpha=0.3)
            #     if len(wec_voltage_data) > 1:
            #         axes[2].legend(loc='upper right')
            # else:
            #     axes[2].text(0.5, 0.5, f"WEC farm bus voltage data\nnot found for buses {wec_buses}", 
            #                 ha='center', va='center', transform=axes[2].transAxes)
            #     axes[2].set_title("WEC-Farm Bus Voltage")
            
            # # Set overall title and adjust layout
            # fig.suptitle(f"WEC Farm Analysis - {software.upper()}", fontsize=16, fontweight='bold')
            # plt.tight_layout()
            # if show:
            #     plt.show()
            
        except Exception as e:
            print(f"Error in WEC analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message on first subplot
            axes[0].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title("WEC Farm Analysis - Error")
            for i in range(1, 3):
                axes[i].text(0.5, 0.5, "Error occurred", ha='center', va='center', transform=axes[i].transAxes)
            if show:
                plt.show()
        
        return fig, axes


    def quick_overview(self, software="pypsa"):
        """Generate a quick overview of all major grid parameters.
        
        Creates a comprehensive set of overview plots for rapid system assessment.
        This method generates multiple plots showing key electrical quantities
        across different component types.
        
        Args:
            software (str, optional): Backend software to use for data.
                Defaults to "psse".
        
        Returns:
            None: Displays plots directly using matplotlib.
        
        Generated Plots:
            1. **Generator Active Power**: All generator outputs [MW]
            2. **Bus Voltages**: System-wide voltage profiles [p.u.]
            3. **Line Loading**: Transmission line loading percentages [%]
        
        Example:
            >>> # Quick overview of PSS®E simulation results
            >>> plotter.quick_overview("psse")
            >>> 
            >>> # Quick overview of PyPSA simulation results
            >>> plotter.quick_overview("pypsa")
        
        Notes:
            - Provides console output indicating which component is being plotted
            - Uses standard plot sizes optimized for overview purposes
            - Automatically handles missing data gracefully
            - Designed for rapid system assessment and validation
            - Calls individual component plotting methods internally
            
        See Also:
            plot_generator: Individual generator plotting
            plot_bus: Individual bus plotting  
            plot_line: Individual line plotting
        """
        print(f"\n=== {software.upper()} Grid Overview ===")
        
        # Generator overview
        print("\n--- Generators ---")
        self.plot_generator(software, "p", figsize=(10, 4))
        
        # Bus overview  
        print("\n--- Bus Voltages ---")
        self.plot_bus(software, "v_mag", figsize=(10, 4))
        
        # Line overview
        print("\n--- Line Loading ---")
        self.plot_line(software, "line_pct", figsize=(10, 4))

    def comparison_suite(self):
        """Run a comprehensive comparison between PSS®E and PyPSA simulation results.
        
        Executes a complete validation suite comparing all major electrical quantities
        between PSS®E and PyPSA simulation results. This method is essential for
        cross-platform validation and model verification.
        
        Returns:
            None: Displays comparison plots and prints statistical metrics to console.
        
        Comparison Categories:
            1. **Generator Comparisons**:
               - Active power output [MW]
               - Reactive power output [MVAr]
            2. **Bus Comparisons**:
               - Voltage magnitudes [p.u.]
               - Voltage angles [degrees]
            3. **Line Comparisons**:
               - Line loading percentages [%]
        
        Example:
            >>> # Run complete PSS®E vs PyPSA validation
            >>> plotter.comparison_suite()
            ===== PSS®E vs PyPSA Comparison Suite =====
            
            --- Generator Active Power ---
            [Displays side-by-side comparison plots]
            Mean Squared Error: 0.045 MW²
            ...
        
        Console Output:
            - Section headers for each comparison category
            - Statistical comparison metrics for each parameter
            - Mean squared error calculations
            - Visual side-by-side comparison plots
            
        Notes:
            - Requires both PSS®E and PyPSA simulation results in engine
            - Automatically handles missing data gracefully
            - Designed for academic publication and validation studies
            - Provides quantitative metrics for model accuracy assessment
            - Todo: Fix legends and add detailed MSE statistics for each parameter
            
        See Also:
            compare_software: Individual parameter comparison method
            
        Validation Requirements:
            Both engine.psse.grid and engine.pypsa.grid must contain:
            - Complete time-series data for all electrical quantities
            - Matching component naming conventions
            - Synchronized simulation time indices
        """
        # todo fix legends
        # todo Mean squared error details for each
        print("\n=== PSS®E vs PyPSA Comparison Suite ===")
        
        # Generator comparisons
        print("\n--- Generator Active Power ---")
        self.compare_software('gen', 'p')
        
        print("\n--- Generator Reactive Power ---")
        self.compare_software('gen', 'q')
        
        # Bus comparisons
        print("\n--- Bus Voltages ---") 
        self.compare_software('bus', 'v_mag')
        
        print("\n--- Bus Angles ---")
        self.compare_software('bus', 'angle_deg')
        
        # Line comparisons
        print("\n--- Line Loading ---")
        self.compare_software('line', 'line_pct')
        
        
    def compare_software(self, component_type='gen', parameter='p', component_name=None, 
                        figsize=(14, 6), max_legend_items=10, **kwargs):
        """Compare identical parameters between PSS®E and PyPSA simulations with statistical analysis.
        
        Creates side-by-side plots comparing the same electrical parameter between
        PSS®E and PyPSA simulation results, along with comprehensive statistical
        metrics printed to console for quantitative validation.
        
        Args:
            component_type (str, optional): Type of component to compare. 
                Supported: 'gen', 'bus', 'line', 'load'. Defaults to 'gen'.
            parameter (str, optional): Electrical parameter to compare.
                Supported: 'p', 'q', 'v_mag', 'angle_deg', 'line_pct', etc.
                Defaults to 'p'.
            component_name (str or list, optional): Specific component(s) to compare.
                If None, all components are included. Defaults to None.
            figsize (tuple, optional): Figure size (width, height). 
                Defaults to (14, 6).
            max_legend_items (int, optional): Maximum legend items before hiding legend.
                Defaults to 10.
            **kwargs: Additional plotting arguments passed to matplotlib.
        
        Returns:
            None: Displays plots and prints statistical metrics to console.
        
        Statistical Metrics Calculated:
            - **MSE**: Mean Squared Error
            - **RMSE**: Root Mean Squared Error  
            - **MAE**: Mean Absolute Error
            - **MAPE**: Mean Absolute Percentage Error [%]
            - **R²**: Coefficient of Determination (correlation²)
        
        Example:
            >>> # Compare all generator active power
            >>> plotter.compare_software('gen', 'p')
            
            >>> # Compare specific bus voltages
            >>> plotter.compare_software('bus', 'v_mag', component_name=['1', '14'])
            
            >>> # Compare line loading with custom figure size
            >>> plotter.compare_software('line', 'line_pct', figsize=(16, 8))
        
        Console Output Example:
            --- Comparison Metrics for GEN P ---
            Found 5 common components for comparison
              Gen_1: MSE=2.3e-03, RMSE=0.048, MAE=0.032, MAPE=1.2%, R²=0.998
              Gen_2: MSE=1.8e-04, RMSE=0.013, MAE=0.009, MAPE=0.8%, R²=0.999
            Overall Statistics:
              Mean MSE: 1.2e-03 ± 8.9e-04
              Mean R²: 0.998 ± 0.001
        
        Plot Features:
            - Intelligent legend management for large datasets
            - Automatic unit labeling based on parameter type
            - Synchronized y-axes for direct comparison
            - Error handling with informative messages
            - Grid lines for enhanced readability
            
        Notes:
            - Requires both PSS®E and PyPSA simulation results in engine
            - Automatically finds common components between software platforms
            - Handles missing data and type conversion gracefully
            - Provides both visual and quantitative validation
            - Essential for cross-platform model verification
            
        Supported Parameter Units:
            - 'p': Active Power [MW]
            - 'q': Reactive Power [MVAr]
            - 'v_mag': Voltage Magnitude [p.u.]
            - 'angle_deg': Voltage Angle [degrees]
            - 'line_pct': Line Loading [%]
            - 'pf': Power Factor [dimensionless]
            - 'freq': Frequency [Hz]
        """
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # Store data for comparison metrics
        psse_data = None
        pypsa_data = None
        
        for i, software in enumerate(['psse', 'pypsa']):
            ax = ax1 if i == 0 else ax2
            
            try:
                grid_obj = getattr(self.engine, software).grid
                component_data = getattr(grid_obj, f"{component_type}_t")
                data = getattr(component_data, parameter)
                
                if component_name:
                    if isinstance(component_name, str):
                        component_name = [component_name]
                    available_components = [c for c in component_name if c in data.columns]
                    if available_components:
                        data = data[available_components]
                    else:
                        print(f"Warning: No specified components found in {software}")
                
                # Store data for metrics calculation
                if software == 'psse':
                    psse_data = data
                else:
                    pypsa_data = data
                
                # Handle legend intelligently
                show_legend = True
                legend_location = 'upper right'
                bbox_to_anchor = None
                
                if isinstance(data, pd.DataFrame):
                    n_components = len(data.columns)
                    
                    # If too many components, limit legend or turn it off
                    if n_components > max_legend_items:
                        show_legend = False
                        # Add text showing number of components
                        ax.text(0.02, 0.98, f"{n_components} components", 
                            transform=ax.transAxes, va='top', ha='left',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    elif n_components > 5:
                        legend_location = 'upper left'
                        bbox_to_anchor = (-0.15, 1)  # Position outside plot area
                    elif n_components > 3:
                        legend_location = 'upper right'
                        # Use smaller font and multiple columns
                
                # Plot the data
                if isinstance(data, pd.Series):
                    ax.plot(data.index, data.values, label=data.name, **kwargs)
                    if show_legend:
                        ax.legend(loc='upper right', fontsize=8)
                else:
                    data.plot(ax=ax, legend=False, **kwargs)  # Turn off default legend
                    
                    # Add custom legend if appropriate
                    if show_legend:
                        if bbox_to_anchor:
                            # Place legend outside plot area for medium datasets
                            legend = ax.legend(bbox_to_anchor=bbox_to_anchor, loc=legend_location, fontsize=7)
                            # Adjust plot to make room for legend
                            plt.subplots_adjust(left=0.2)
                        else:
                            # Standard legend for small datasets
                            ncols = min(2, max(1, n_components // 6))  # Dynamic column count
                            ax.legend(loc=legend_location, fontsize=8, ncol=ncols)
                
                ax.set_title(f"{software.upper()}", fontsize=12, fontweight='bold')
                ax.set_xlabel("Time")
                ax.grid(True, alpha=0.3)
                
                # Set y-label based on parameter
                ylabel_map = {
                    'p': 'Active Power (MW)',
                    'q': 'Reactive Power (MVAr)', 
                    'v': 'Voltage (p.u.)',
                    'v_mag': 'Voltage Magnitude (p.u.)',
                    'angle_deg': 'Voltage Angle (degrees)',
                    'line_pct': 'Line Loading (%)',
                    'pf': 'Power Factor',
                    'freq': 'Frequency (Hz)'
                }
                ax.set_ylabel(ylabel_map.get(parameter, parameter.upper()))
                
            except Exception as e:
                print(f"Error plotting {software}: {e}")
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=ax.transAxes)
        
        # Calculate and print comparison metrics to console
        try:
            if psse_data is not None and pypsa_data is not None:
                print(f"\n--- Comparison Metrics for {component_type.upper()} {parameter.upper()} ---")
                
                # Handle DataFrame comparison
                if isinstance(psse_data, pd.DataFrame) and isinstance(pypsa_data, pd.DataFrame):
                    # Find common columns
                    common_cols = list(set(psse_data.columns) & set(pypsa_data.columns))
                    if common_cols:
                        print(f"Found {len(common_cols)} common components for comparison")
                        
                        overall_metrics = {'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []}
                        
                        for col in common_cols[:10]:  # Show metrics for first 10 components
                            try:
                                # Get time series data correctly
                                p_series = psse_data[col]
                                py_series = pypsa_data[col]
                                
                                # Check if we have proper time series data
                                if isinstance(p_series, pd.Series) and isinstance(py_series, pd.Series):
                                    # Convert to numeric and handle object dtype
                                    p_series_clean = pd.to_numeric(p_series, errors='coerce').dropna()
                                    py_series_clean = pd.to_numeric(py_series, errors='coerce').dropna()
                                    
                                    # Get values as numpy arrays
                                    p_vals = p_series_clean.values.astype(float)
                                    py_vals = py_series_clean.values.astype(float)
                                    
                                    # Ensure same length
                                    min_len = min(len(p_vals), len(py_vals))
                                    if min_len < 2:  # Need at least 2 points for meaningful comparison
                                        print(f"  {col}: Insufficient data points ({min_len})")
                                        continue
                                        
                                    p_vals = p_vals[:min_len]
                                    py_vals = py_vals[:min_len]
                                    
                                    # Calculate metrics
                                    mse = np.mean((p_vals - py_vals)**2)
                                    rmse = np.sqrt(mse)
                                    mae = np.mean(np.abs(p_vals - py_vals))
                                    
                                    # Avoid division by zero for MAPE
                                    p_abs_mean = np.mean(np.abs(p_vals))
                                    if p_abs_mean > 1e-10:
                                        mape = np.mean(np.abs((p_vals - py_vals) / np.maximum(np.abs(p_vals), 1e-10))) * 100
                                    else:
                                        mape = 0
                                    
                                    # Correlation coefficient
                                    if np.std(p_vals) > 1e-10 and np.std(py_vals) > 1e-10:
                                        corr = np.corrcoef(p_vals, py_vals)[0, 1]
                                        r2 = corr**2
                                    else:
                                        r2 = 0
                                    
                                    print(f"  {col}: MSE={mse:.2e}, RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.1f}%, R²={r2:.3f}")
                                    
                                    # Store for overall statistics
                                    overall_metrics['mse'].append(mse)
                                    overall_metrics['rmse'].append(rmse)
                                    overall_metrics['mae'].append(mae)
                                    overall_metrics['mape'].append(mape)
                                    overall_metrics['r2'].append(r2)
                                    
                                else:
                                    print(f"  {col}: Data type issue - not time series")
                                    
                            except Exception as e:
                                print(f"  {col}: Error - {str(e)[:50]}")
                                continue
                        
                        # Print overall statistics
                        if overall_metrics['mse']:
                            print(f"\nOverall Statistics ({len(overall_metrics['mse'])} components):")
                            print(f"  Mean MSE: {np.mean(overall_metrics['mse']):.2e}")
                            print(f"  Mean RMSE: {np.mean(overall_metrics['rmse']):.3f}")
                            print(f"  Mean MAE: {np.mean(overall_metrics['mae']):.3f}")
                            print(f"  Mean MAPE: {np.mean(overall_metrics['mape']):.1f}%")
                            print(f"  Mean R²: {np.mean(overall_metrics['r2']):.3f}")
                            
                        if len(common_cols) > 10:
                            print(f"(Showing metrics for first 10 of {len(common_cols)} components)")
                    else:
                        print("No common components found for comparison")
                        
                # Handle Series comparison
                elif isinstance(psse_data, pd.Series) and isinstance(pypsa_data, pd.Series):
                    try:
                        # Convert to numeric and handle object dtype
                        psse_clean = pd.to_numeric(psse_data, errors='coerce').dropna()
                        pypsa_clean = pd.to_numeric(pypsa_data, errors='coerce').dropna()
                        
                        p_vals = psse_clean.values.astype(float)
                        py_vals = pypsa_clean.values.astype(float)
                        
                        min_len = min(len(p_vals), len(py_vals))
                        if min_len > 1:
                            p_vals = p_vals[:min_len]
                            py_vals = py_vals[:min_len]
                            
                            # Calculate metrics
                            mse = np.mean((p_vals - py_vals)**2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(p_vals - py_vals))
                            
                            p_abs_mean = np.mean(np.abs(p_vals))
                            if p_abs_mean > 1e-10:
                                mape = np.mean(np.abs((p_vals - py_vals) / np.maximum(np.abs(p_vals), 1e-10))) * 100
                            else:
                                mape = 0
                            
                            if np.std(p_vals) > 1e-10 and np.std(py_vals) > 1e-10:
                                corr = np.corrcoef(p_vals, py_vals)[0, 1]
                                r2 = corr**2
                            else:
                                r2 = 0
                            
                            print(f"Single component comparison:")
                            print(f"  MSE: {mse:.2e}")
                            print(f"  RMSE: {rmse:.3f}")
                            print(f"  MAE: {mae:.3f}")
                            print(f"  MAPE: {mape:.1f}%")
                            print(f"  R²: {r2:.3f}")
                            print(f"  Correlation: {corr:.3f}")
                            print(f"  Data Points: {min_len}")
                        else:
                            print("Insufficient data points for comparison")
                    except Exception as e:
                        print(f"Metrics calculation error: {e}")
                else:
                    print("Data type mismatch between software")
            else:
                print("No data available for comparison")
                        
        except Exception as e:
            print(f"Error in metrics calculation: {e}")        # Set overall title
        fig.suptitle(f"{component_type.upper()} {parameter.upper()} Comparison", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        #plt.show()
        
        return fig, (ax1, ax2)

    def sld(self, software: str = 'pypsa', figsize=(14, 10), title=None, save_path=None, show=False):
        """Generate single-line diagram using GridState data.
        
        Creates a single-line diagram visualization using the standardized GridState
        component data. Works with both PSS®E and PyPSA backends by using the unified
        data schema from GridState snapshots.
        
        Args:
            software (str): Backend software ("psse" or "pypsa")
            figsize (tuple): Figure size as (width, height)
            title (str, optional): Custom title for the diagram
            save_path (str, optional): Path to save the figure
            show (bool): Whether to display the figure (default: False)
            
        Returns:
            matplotlib.figure.Figure: The generated SLD figure
            
        Notes:
            Uses NetworkX for automatic layout calculation since GridState doesn't
            include geographical bus positions. The diagram includes:
            
            - Buses: Colored rectangles based on type (Slack=red, PV=green, PQ=gray)
            - Lines: Black dashed lines connecting buses
            - Generators: Circles above buses with generators
            - Loads: Downward arrows on buses with loads
            
            Limitations:
            - No transformer identification (would need additional data)
            - Layout is algorithmic, not geographical
            - No shunt devices (not in GridState schema)
        """
        try:
            # Get the appropriate grid object
            grid_obj = getattr(self.engine, software).grid
        except AttributeError:
            raise ValueError(f"Engine does not have {software} attribute or grid is not loaded")
        
        # Extract data from GridState
        bus_df = grid_obj.bus.copy()
        line_df = grid_obj.line.copy()
        gen_df = grid_obj.gen.copy()
        load_df = grid_obj.load.copy()
        
        if bus_df.empty:
            raise ValueError("No bus data available for SLD generation")
        
        print(f"SLD Data Summary:")
        print(f"  Buses: {len(bus_df)}")
        print(f"  Lines: {len(line_df)}")
        print(f"  Generators: {len(gen_df)}")
        print(f"  Loads: {len(load_df)}")
        
        # Check if required columns exist
        if 'bus' not in bus_df.columns and bus_df.index.name != 'bus':
            print(f"  ERROR: 'bus' column missing from bus DataFrame")
            print(f"  Available columns: {list(bus_df.columns)}")
            print(f"  Index name: {bus_df.index.name}")
            print(f"  Bus DataFrame head:\n{bus_df.head()}")
            
            # Check if bus numbers are in the index
            if bus_df.index.name == 'bus' or 'bus' in str(bus_df.index.name).lower():
                print("  Bus numbers found in DataFrame index, will use index values")
            else:
                raise ValueError("Bus DataFrame missing required 'bus' column or index")
        
        # Create network graph for layout
        G = nx.Graph()
        
        # Add buses as nodes - handle index vs column
        if 'bus' in bus_df.columns:
            bus_numbers = bus_df['bus']
        else:
            # Bus numbers are in the index
            bus_numbers = bus_df.index
            
        for bus_num in bus_numbers:
            G.add_node(bus_num)
        
        # Add lines as edges - handle potential column name variations  
        ibus_col = 'ibus' if 'ibus' in line_df.columns else 'from_bus'
        jbus_col = 'jbus' if 'jbus' in line_df.columns else 'to_bus'
        status_col = 'status' if 'status' in line_df.columns else None
        
        for _, line_row in line_df.iterrows():
            if status_col is None or line_row[status_col] == 1:  # Only active lines
                if ibus_col in line_df.columns and jbus_col in line_df.columns:
                    G.add_edge(line_row[ibus_col], line_row[jbus_col])
        
        # Calculate layout using NetworkX
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            # Fallback to spring layout if kamada_kawai fails
            pos = nx.spring_layout(G, seed=42)
        
        # Normalize positions for better visualization
        if pos:
            pos_values = np.array(list(pos.values()))
            x_vals, y_vals = pos_values[:, 0], pos_values[:, 1]
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            
            # Normalize to reasonable plotting range
            for node in pos:
                pos[node] = (
                    2 * (pos[node][0] - x_min) / (x_max - x_min) - 1,
                    1.5 * (pos[node][1] - y_min) / (y_max - y_min) - 0.5
                )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bus visualization parameters
        node_width, node_height = 0.12, 0.04
        
        # Bus type color mapping
        bus_colors = {
            "Slack": "#FF4500",  # Red-orange
            "PV": "#32CD32",     # Green  
            "PQ": "#A9A9A9",     # Gray
        }
        
        # Draw transmission lines first (so they appear behind buses)
        for _, line_row in line_df.iterrows():
            if status_col is None or line_row[status_col] == 1:  # Only active lines
                if ibus_col in line_df.columns and jbus_col in line_df.columns:
                    ibus, jbus = line_row[ibus_col], line_row[jbus_col]
                    if ibus in pos and jbus in pos:
                        x1, y1 = pos[ibus]
                        x2, y2 = pos[jbus]
                        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.7)
        
        # Identify buses with generators and loads - handle column variations
        gen_bus_col = 'bus' if 'bus' in gen_df.columns else 'connected_bus'
        load_bus_col = 'bus' if 'bus' in load_df.columns else 'connected_bus'
        gen_status_col = 'status' if 'status' in gen_df.columns else None
        load_status_col = 'status' if 'status' in load_df.columns else None
        
        # Get active generators and loads
        if gen_status_col:
            gen_buses = set(gen_df[gen_df[gen_status_col] == 1][gen_bus_col])
        else:
            gen_buses = set(gen_df[gen_bus_col])
            
        if load_status_col:
            load_buses = set(load_df[load_df[load_status_col] == 1][load_bus_col])
        else:
            load_buses = set(load_df[load_bus_col])
        
        # Draw buses
        bus_type_col = 'type' if 'type' in bus_df.columns else 'control'
        # Determine bus column name
        if 'bus' in bus_df.columns:
            bus_col = 'bus'
        else:
            # Bus numbers are in the index
            bus_col = None
            
        for _, bus_row in bus_df.iterrows():
            if bus_col:
                bus_num = bus_row[bus_col]
            else:
                bus_num = bus_row.name  # Use index value
            if bus_num not in pos:
                continue
                
            x, y = pos[bus_num]
            bus_type = bus_row[bus_type_col] if bus_type_col in bus_df.columns else "PQ"
            bus_color = bus_colors.get(bus_type, "#D3D3D3")  # Default light gray
            
            # Draw bus rectangle
            rect = Rectangle(
                (x - node_width / 2, y - node_height / 2), 
                node_width, node_height,
                linewidth=1.5, 
                edgecolor='black', 
                facecolor=bus_color
            )
            ax.add_patch(rect)
            
            # Add bus number label
            ax.text(x, y, str(bus_num), fontsize=8, fontweight="bold", 
                   ha='center', va='center')
            
            # Draw generators (circles above bus)
            if bus_num in gen_buses:
                gen_x = x
                gen_y = y + node_height / 2 + 0.05
                gen_size = 0.02
                # Connection line from bus to generator
                ax.plot([x, gen_x], [y + node_height / 2, gen_y - gen_size], 
                       color='black', linewidth=2)
                # Generator circle
                ax.add_patch(Circle((gen_x, gen_y), gen_size, 
                                  color='none', ec='black', linewidth=1.5))
                # Generator symbol 'G'
                ax.text(gen_x, gen_y, 'G', fontsize=6, fontweight="bold",
                       ha='center', va='center')
            
            # Draw loads (downward arrows)
            if bus_num in load_buses:
                load_x = x + node_width / 2 - 0.02
                load_y = y - node_height / 2
                ax.arrow(load_x, load_y, 0, -0.04, 
                        head_width=0.015, head_length=0.015, 
                        fc='black', ec='black')
        
        # Set up the plot
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        
        # Set title
        if title is None:
            case_name = getattr(self.engine, 'case_name', 'Power System')
            title = f"Single-Line Diagram - {case_name} ({software.upper()})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='black', markersize=8, 
                   label="Generator", markerfacecolor='none', 
                   markeredgecolor='black', linewidth=0),
            Line2D([0], [0], marker='^', color='black', markersize=8, 
                   label="Load", markerfacecolor='black', linewidth=0),
            Line2D([0], [0], marker='s', color='#FF4500', markersize=8, 
                   label="Slack Bus", markerfacecolor='#FF4500', linewidth=0),
            Line2D([0], [0], marker='s', color='#32CD32', markersize=8, 
                   label="PV Bus", markerfacecolor='#32CD32', linewidth=0),
            Line2D([0], [0], marker='s', color='#A9A9A9', markersize=8, 
                   label="PQ Bus", markerfacecolor='#A9A9A9', linewidth=0),
            Line2D([0], [0], color='black', linewidth=1.5, 
                   label="Transmission Line"),
        ]
        
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10, 
                 frameon=True, edgecolor='black', title="Legend")
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SLD saved to: {save_path}")
        
        plt.tight_layout()
        
        # Only show if explicitly requested
        if show:
            plt.show()
        
        return fig