"""
WEC-Grid high-level plotting interface

This module provides comprehensive visualization capabilities for WEC-GRID simulation
results, supporting cross-platform comparison between PSS®E and PyPSA modeling backends.
"""

import matplotlib.pyplot as plt
import pandas as pd


class WECGridPlotter:
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
    
    def __init__(self, engine):
        """Initialize WECGridPlotter with WEC-GRID Engine.
        
        Args:
            engine: WEC-GRID Engine with 'psse' and/or 'pypsa' attributes.
        """
        self.engine = engine
    
    def plot_component_data(self, software, component_type, parameter, component_name=None, 
                           figsize=(12, 6), style='default', show_grid=True, 
                           save_path=None, **plot_kwargs):
        """Plot time series data for grid components.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            component_type (str): Component type ("gen", "bus", "line", "load").
            parameter (str): Parameter to plot ("p", "q", "v_mag", "angle_deg", "line_pct").
            component_name (str or list, optional): Specific component(s) to plot.
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
            >>> # Plot specific bus voltages
            >>> plotter.plot_component_data("pypsa", "bus", "v_mag", 
            ...                             component_name=["1", "2", "3"])
            
        Notes:
            - Automatically handles data validation and missing components
            - Y-axis labels set based on parameter type
            - Legend managed intelligently for multiple components
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
        
        # Handle component selection
        if component_name is not None:
            if isinstance(component_name, str):
                component_name = [component_name]
            
            # Check if components exist
            missing_components = [comp for comp in component_name if comp not in data.columns]
            if missing_components:
                print(f"Warning: Components not found: {missing_components}")
                component_name = [comp for comp in component_name if comp in data.columns]
            
            if not component_name:
                print("No valid components found to plot")
                return None, None
            
            data = data[component_name]
        
        # Add data summary
        if isinstance(data, pd.DataFrame):
            print(f"Plotting {len(data.columns)} components over {len(data)} time steps")
            if data.empty:
                print("Warning: No data to plot")
                return None, None
        else:
            print(f"Plotting single component over {len(data)} time steps")
        
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
        if component_name:
            title += f" ({', '.join(component_name) if len(component_name) <= 3 else f'{len(component_name)} components'})"
        
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
        
        plt.show()
        
        return fig, ax

    def plot_generator(self, software, parameter='p', component_name=None, **kwargs):
        """Plot generator time-series data.
        
        Convenience method for plotting generator parameters using the main
        plot_component_data method with component_type='gen'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Generator parameter to plot. Defaults to 'p'.
                Common values: 'p' (active power), 'q' (reactive power), 'status'.
            component_name (str or list, optional): Specific generator(s) to plot.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all generator active power
            >>> plotter.plot_generator("psse", "p")
            >>> # Plot specific generators' reactive power
            >>> plotter.plot_generator("pypsa", "q", component_name=["1_1", "2_1"])
        """
        return self.plot_component_data(software, 'gen', parameter, component_name, **kwargs)

    def plot_bus(self, software, parameter='p', component_name=None, **kwargs):
        """Plot bus time-series data.
        
        Convenience method for plotting bus parameters using the main
        plot_component_data method with component_type='bus'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Bus parameter to plot. Defaults to 'p'.
                Common values: 'p' (net power), 'q' (net reactive power), 
                'v_mag' (voltage magnitude), 'angle_deg' (voltage angle).
            component_name (str or list, optional): Specific bus(es) to plot.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all bus voltage magnitudes
            >>> plotter.plot_bus("psse", "v_mag")
            >>> # Plot specific bus power injections
            >>> plotter.plot_bus("pypsa", "p", component_name=[1, 2, 14])
        """
        return self.plot_component_data(software, 'bus', parameter, component_name, **kwargs)

    def plot_line(self, software, parameter='line_pct', component_name=None, **kwargs):
        """Plot transmission line time-series data.
        
        Convenience method for plotting line parameters using the main
        plot_component_data method with component_type='line'.
        
        Args:
            software (str): Backend software ("psse" or "pypsa").
            parameter (str, optional): Line parameter to plot. Defaults to 'line_pct'.
                Common values: 'line_pct' (loading percentage), 'status'.
            component_name (str or list, optional): Specific line(s) to plot.
            **kwargs: Additional arguments passed to plot_component_data.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
            
        Example:
            >>> # Plot all line loading percentages
            >>> plotter.plot_line("psse", "line_pct")
            >>> # Plot specific line status
            >>> plotter.plot_line("pypsa", "status", component_name=["Line_1_2_1"])
        """
        return self.plot_component_data(software, 'line', parameter, component_name, **kwargs)

    def plot_load(self, software, parameter='p', component_name=None, **kwargs):
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
        return self.plot_component_data(software, 'load', parameter, component_name, **kwargs)

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
        plt.show()
        return fig, axes


    def plot_wec_analysis(self, software="psse", figsize=(12, 15)):
        """Generate comprehensive WEC farm analysis plots.
        
        Creates a specialized three-panel analysis plot focusing on Wave Energy Converter
        (WEC) farm performance and grid integration effects. This method provides
        publication-quality visualizations for WEC integration studies.
        
        Args:
            software (str, optional): Backend software to use. Defaults to "psse".
            figsize (tuple, optional): Figure size (width, height). Defaults to (12, 15).
        
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
            - Automatically detects WEC farms from engine.wec_farms
            - Uses farm names for clear identification in legends
            - Calculates meaningful metrics like contribution percentages
            - Handles missing data gracefully with informative error messages
            - Provides console output showing detected WEC farms
            - Designed for publication-quality WEC integration analysis
            
        WEC Farm Requirements:
            Each WEC farm object must have:
            - id: Unique identifier for the farm
            - bus_location: Bus number where farm is connected
            - farm_name: Human-readable farm name for plots
        """
        if not hasattr(self.engine, 'wec_farms') or not self.engine.wec_farms:
            print("No WEC farms found in engine")
            return None, None
        
        # Create a figure with 3 subplots (3x1 grid)
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        try:
            # Get grid data
            grid_obj = getattr(self.engine, software).grid
            gen_data = getattr(grid_obj, 'gen_t')
            bus_data = getattr(grid_obj, 'bus_t')
            
            # Collect WEC farm information
            wec_farm_info = []
            for farm in self.engine.wec_farms:
                if hasattr(farm, 'id') and hasattr(farm, 'bus_location') and hasattr(farm, 'farm_name'):
                    wec_gen_name = f"{farm.bus_location}_{farm.id}"
                    wec_farm_info.append({
                        'gen_name': wec_gen_name,
                        'bus_location': farm.bus_location,
                        'farm_name': farm.farm_name
                    })
            
            print(f"Found {len(wec_farm_info)} WEC farms:")
            for info in wec_farm_info:
                print(f"  - {info['farm_name']}: {info['gen_name']} at bus {info['bus_location']}")
            
            # Plot 1: WEC-Farm Active Power Output
            p_data = getattr(gen_data, 'p')
            wec_generators = [info['gen_name'] for info in wec_farm_info]
            available_wec_gens = [gen for gen in wec_generators if gen in p_data.columns]
            
            if available_wec_gens:
                wec_p_data = p_data[available_wec_gens]
                
                # Create labels using farm names
                plot_data = wec_p_data.copy()
                # Rename columns to use farm names instead of generator IDs
                column_mapping = {}
                for info in wec_farm_info:
                    if info['gen_name'] in plot_data.columns:
                        column_mapping[info['gen_name']] = info['farm_name']
                plot_data.rename(columns=column_mapping, inplace=True)
                
                plot_data.plot(ax=axes[0], linewidth=2, title="WEC-Farm Active Power Output")
                axes[0].set_ylabel("Active Power (MW)")
                axes[0].grid(True, alpha=0.3)
                if len(available_wec_gens) > 1:
                    axes[0].legend(loc='upper right')
            else:
                axes[0].text(0.5, 0.5, "No WEC farm generators\nfound in data", 
                            ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title("WEC-Farm Active Power Output")
                
            # Plot 2: WEC Contribution Percentage over Time
            if available_wec_gens:
                wec_p_data = p_data[available_wec_gens]
                total_wec_power = wec_p_data.sum(axis=1)
                total_gen_power = p_data.sum(axis=1)
                
                # Calculate WEC percentage of total generation
                wec_percentage = (total_wec_power / total_gen_power) * 100
                
                axes[1].plot(wec_percentage.index, wec_percentage.values, 
                            label="WEC Contribution %", linewidth=2, color='green')
                axes[1].set_title("WEC-Farm Contribution to Total Generation (%)")
                axes[1].set_ylabel("Percentage (%)")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Plot 3: WEC-Farm Bus Voltage
                v_data = getattr(bus_data, 'v_mag')
                wec_buses = [info['bus_location'] for info in wec_farm_info]
                available_wec_buses = [bus for bus in wec_buses if bus in v_data.columns]
            
            # Plot 3: WEC-Farm Bus Voltage
            v_data = getattr(bus_data, 'v_mag')
            wec_buses = [info['bus_location'] for info in wec_farm_info]
            available_wec_buses = [bus for bus in wec_buses if bus in v_data.columns]
            
            if available_wec_buses:
                wec_bus_v_data = v_data[available_wec_buses]
                
                # Create labels using farm names and bus numbers
                plot_data = wec_bus_v_data.copy()
                column_mapping = {}
                for info in wec_farm_info:
                    if info['bus_location'] in plot_data.columns:
                        column_mapping[info['bus_location']] = f"{info['farm_name']} (Bus {info['bus_location']})"
                plot_data.rename(columns=column_mapping, inplace=True)
                
                plot_data.plot(ax=axes[2], linewidth=2, title="WEC-Farm Bus Voltage")
                axes[2].set_ylabel("Voltage (p.u.)")
                axes[2].grid(True, alpha=0.3)
                if len(available_wec_buses) > 1:
                    axes[2].legend(loc='upper right')
            else:
                axes[2].text(0.5, 0.5, f"WEC farm buses\n{wec_buses}\nnot found in voltage data", 
                            ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title("WEC-Farm Bus Voltage")
            
            # Set overall title and adjust layout
            fig.suptitle(f"WEC Farm Analysis - {software.upper()}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in WEC analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message on first subplot
            axes[0].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title("WEC Farm Analysis - Error")
            for i in range(1, 3):
                axes[i].text(0.5, 0.5, "Error occurred", ha='center', va='center', transform=axes[i].transAxes)
            plt.show()
        
        return fig, axes


    def quick_overview(self, software="psse"):
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
        plt.show()
        
        return fig, (ax1, ax2)