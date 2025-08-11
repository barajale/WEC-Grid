"""
WEC-Grid high-level plotting interface
"""

import matplotlib.pyplot as plt
import pandas as pd


class WECGridPlotter:
    def __init__(self, engine):
        """
        Args:
            engine: the WEC-Grid Engine instance
        """
        self.engine = engine
    
    def plot_component_data(self, software, component_type, parameter, component_name=None, 
                           figsize=(12, 6), style='default', show_grid=True, 
                           save_path=None, **plot_kwargs):
        """
        Plot time series data for grid components from PSS®E or PyPSA simulations.
        
        Parameters:
        -----------
        software : str
            Either "psse" or "pypsa"
        component_type : str
            Component type: "gen", "bus", "line", "load"
        parameter : str
            Parameter to plot: "p", "q", "v", etc.
        component_name : str or list, optional
            Specific component(s) to plot. If None, plots all components
        figsize : tuple
            Figure size (width, height)
        style : str
            Matplotlib style (use plt.style.available to see options)
        show_grid : bool
            Whether to show grid lines
        save_path : str, optional
            Path to save the figure
        **plot_kwargs : dict
            Additional plotting arguments (linewidth, alpha, etc.)
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
        """Plot generator data"""
        return self.plot_component_data(software, 'gen', parameter, component_name, **kwargs)

    def plot_bus(self, software, parameter='p', component_name=None, **kwargs):
        """Plot bus data"""
        return self.plot_component_data(software, 'bus', parameter, component_name, **kwargs)

    def plot_line(self, software, parameter='line_pct', component_name=None, **kwargs):
        """Plot line data"""
        return self.plot_component_data(software, 'line', parameter, component_name, **kwargs)

    def plot_load(self, software, parameter='p', component_name=None, **kwargs):
        """Plot load data"""
        return self.plot_component_data(software, 'load', parameter, component_name, **kwargs)

    def compare_software(self, component_type='gen', parameter='p', component_name=None, 
                        figsize=(14, 6), **kwargs):
        """Compare the same parameter between PSS®E and PyPSA side-by-side"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        for i, software in enumerate(['psse', 'pypsa']):
            ax = ax1 if i == 0 else ax2
            
            try:
                grid_obj = getattr(self.engine, software).grid
                component_data = getattr(grid_obj, f"{component_type}_t")
                data = getattr(component_data, parameter)
                
                if component_name:
                    if isinstance(component_name, str):
                        component_name = [component_name]
                    data = data[component_name] if any(c in data.columns for c in component_name) else data
                
                if isinstance(data, pd.Series):
                    ax.plot(data.index, data.values, **kwargs)
                else:
                    data.plot(ax=ax, **kwargs)
                
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
        
        # Set overall title
        fig.suptitle(f"{component_type.upper()} {parameter.upper()} Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return fig, (ax1, ax2)

    def plot_component_grid(self, software, component_type, parameters, 
                           component_name=None, figsize=(15, 10), **kwargs):
        """Plot multiple parameters in a grid"""
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


    def plot_wec_analysis(self, software="psse", figsize=(16, 12)):
        """Comprehensive WEC analysis plotting"""
        if not hasattr(self.engine, 'wec_farms') or not self.engine.wec_farms:
            print("No WEC farms found in engine")
            return None, None
        
        # Create a figure with 6 subplots (2x3 grid)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        try:
            # Get WEC generator names from all farms using farm.id and farm.bus_location
            wec_gen_names = []
            for farm in self.engine.wec_farms:
                if hasattr(farm, 'id') and hasattr(farm, 'bus_location'):
                    wec_gen_name = f"{farm.bus_location}_{farm.id}"
                    wec_gen_names.append(wec_gen_name)
            
            print(f"Looking for WEC generators: {wec_gen_names}")
            
            # Get grid data
            grid_obj = getattr(self.engine, software).grid
            gen_data = getattr(grid_obj, 'gen_t')
            
            # Plot 1: WEC Active Power Output
            if wec_gen_names:
                p_data = getattr(gen_data, 'p')
                
                # Filter for WEC generators only
                available_wecs = [wec for wec in wec_gen_names if wec in p_data.columns]
                print(f"Available WEC generators in data: {available_wecs}")
                
                if available_wecs:
                    wec_p_data = p_data[available_wecs]
                    wec_p_data.plot(ax=axes[0], title="WEC Active Power Output", linewidth=2)
                    axes[0].set_ylabel("Active Power (MW)")
                    axes[0].grid(True, alpha=0.3)
                    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[0].text(0.5, 0.5, "No WEC generators\nfound in data", 
                            ha='center', va='center', transform=axes[0].transAxes)
            
            # Plot 2: WEC Reactive Power Output
            if wec_gen_names and available_wecs:
                q_data = getattr(gen_data, 'q')
                available_wecs_q = [wec for wec in wec_gen_names if wec in q_data.columns]
                if available_wecs_q:
                    wec_q_data = q_data[available_wecs_q]
                    wec_q_data.plot(ax=axes[1], title="WEC Reactive Power Output", linewidth=2)
                    axes[1].set_ylabel("Reactive Power (MVAr)")
                    axes[1].grid(True, alpha=0.3)
                    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 3: Bus voltage at WEC location
            # each farm can be at a different bus, we can get the bus location for each bus by doing farm.bus_location
            wec_bus = getattr(self.engine.wec_farms[0], 'bus_location', None)
            if wec_bus:
                bus_data = getattr(grid_obj, 'bus_t')
                v_data = getattr(bus_data, 'v_mag')
                if str(wec_bus) in v_data.columns:
                    v_data[str(wec_bus)].plot(ax=axes[2], title=f"Voltage at WEC Bus {wec_bus}", linewidth=2)
                    axes[2].set_ylabel("Voltage (p.u.)")
                    axes[2].grid(True, alpha=0.3)
                else:
                    axes[2].text(0.5, 0.5, f"Bus {wec_bus} not found\nin voltage data", 
                            ha='center', va='center', transform=axes[2].transAxes)
            
            # Plot 4: Total WEC Power vs Total Grid Generation
            if wec_gen_names and available_wecs:
                p_data = getattr(gen_data, 'p')
                wec_p_data = p_data[available_wecs]
                total_wec_power = wec_p_data.sum(axis=1)
                total_gen_power = p_data.sum(axis=1)
                
                axes[3].plot(total_wec_power.index, total_wec_power.values, 
                            label="Total WEC Power", linewidth=2)
                axes[3].plot(total_gen_power.index, total_gen_power.values, 
                            label="Total Grid Generation", linewidth=2)
                axes[3].set_title("WEC vs Total Generation")
                axes[3].set_ylabel("Active Power (MW)")
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
            
            # Plot 5: WEC Power Factor (if available)
            try:
                if wec_gen_names and available_wecs:
                    p_data = getattr(gen_data, 'p')
                    q_data = getattr(gen_data, 'q')
                    wec_p_data = p_data[available_wecs]
                    wec_q_data = q_data[available_wecs]
                    
                    # Calculate power factor: P / sqrt(P^2 + Q^2)
                    wec_pf = wec_p_data / (wec_p_data**2 + wec_q_data**2)**0.5
                    wec_pf.plot(ax=axes[4], title="WEC Power Factor", linewidth=2)
                    axes[4].set_ylabel("Power Factor")
                    axes[4].grid(True, alpha=0.3)
                    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            except Exception as e:
                axes[4].text(0.5, 0.5, f"Power Factor\nCalculation Error\n{str(e)}", 
                            ha='center', va='center', transform=axes[4].transAxes)
            
            # Plot 6: WEC Efficiency or Utilization
            try:
                if wec_gen_names and available_wecs:
                    p_data = getattr(gen_data, 'p')
                    wec_p_data = p_data[available_wecs]
                    
                    # Show individual WEC contributions as percentage of total WEC power
                    total_wec = wec_p_data.sum(axis=1)
                    wec_percentage = wec_p_data.div(total_wec, axis=0) * 100
                    wec_percentage.plot(ax=axes[5], title="Individual WEC Contributions (%)", linewidth=2)
                    axes[5].set_ylabel("Percentage of Total WEC Power (%)")
                    axes[5].grid(True, alpha=0.3)
                    axes[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            except Exception as e:
                axes[5].text(0.5, 0.5, f"WEC Contribution\nCalculation Error\n{str(e)}", 
                            ha='center', va='center', transform=axes[5].transAxes)
            
            # Set overall title
            fig.suptitle(f"WEC Farm Analysis - {software.upper()}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in WEC analysis: {e}")
            # Show error message on first subplot
            axes[0].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=axes[0].transAxes)
            plt.show()
        
        return fig, axes


    def quick_overview(self, software="psse"):
        """Generate a quick overview of all major grid parameters"""
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
        """Run a full comparison between PSS®E and PyPSA"""
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