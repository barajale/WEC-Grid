import os
import numpy as np
import wecgrid 
import json



# Compare timing data between WEC and non-WEC systems
import pandas as pd
import matplotlib.pyplot as plt

def compare_timing_data(time_data_1, time_data_2, label_1="System 1", label_2="System 2"):
    """Compare timing data between two systems."""
    
    print(f"\n{'='*80}")
    print(f"TIMING COMPARISON: {label_1} vs {label_2}")
    print(f"{'='*80}")
    
    # Create comparison DataFrame
    comparison_data = []
    
    # Total simulation time
    total_1 = time_data_1.get('simulation_total', 0)
    total_2 = time_data_2.get('simulation_total', 0)
    
    # Average times for key operations
    avg_pf_1 = np.mean(time_data_1.get('solve_powerflow_times', [0])) * 1000  # Convert to ms
    avg_pf_2 = np.mean(time_data_2.get('solve_powerflow_times', [0])) * 1000
    
    avg_snap_1 = np.mean(time_data_1.get('take_snapshot_times', [0])) * 1000
    avg_snap_2 = np.mean(time_data_2.get('take_snapshot_times', [0])) * 1000
    
    avg_iter_1 = np.mean(time_data_1.get('iteration_times', [0])) * 1000
    avg_iter_2 = np.mean(time_data_2.get('iteration_times', [0])) * 1000
    
    # Print detailed comparison
    print(f"{'Metric':<25} | {label_1:<15} | {label_2:<15} | {'Difference':<15} | {'% Change':<10}")
    print(f"{'-'*25} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*10}")
    
    # Total simulation time
    diff_total = total_2 - total_1
    pct_total = (diff_total / total_1 * 100) if total_1 > 0 else 0
    print(f"{'Total Simulation (s)':<25} | {total_1:<15.3f} | {total_2:<15.3f} | {diff_total:<15.3f} | {pct_total:<10.1f}%")
    
    # Average power flow time
    diff_pf = avg_pf_2 - avg_pf_1
    pct_pf = (diff_pf / avg_pf_1 * 100) if avg_pf_1 > 0 else 0
    print(f"{'Avg PowerFlow (ms)':<25} | {avg_pf_1:<15.2f} | {avg_pf_2:<15.2f} | {diff_pf:<15.2f} | {pct_pf:<10.1f}%")
    
    # Average snapshot time
    diff_snap = avg_snap_2 - avg_snap_1
    pct_snap = (diff_snap / avg_snap_1 * 100) if avg_snap_1 > 0 else 0
    print(f"{'Avg Snapshot (ms)':<25} | {avg_snap_1:<15.2f} | {avg_snap_2:<15.2f} | {diff_snap:<15.2f} | {pct_snap:<10.1f}%")
    
    # Average iteration time
    diff_iter = avg_iter_2 - avg_iter_1
    pct_iter = (diff_iter / avg_iter_1 * 100) if avg_iter_1 > 0 else 0
    print(f"{'Avg Iteration (ms)':<25} | {avg_iter_1:<15.2f} | {avg_iter_2:<15.2f} | {diff_iter:<15.2f} | {pct_iter:<10.1f}%")
    
    # Number of iterations
    n_iter_1 = len(time_data_1.get('iteration_times', []))
    n_iter_2 = len(time_data_2.get('iteration_times', []))
    print(f"{'Number of Iterations':<25} | {n_iter_1:<15} | {n_iter_2:<15} | {n_iter_2-n_iter_1:<15} | {'-':<10}")
    
    print(f"{'='*80}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Timing Comparison: {label_1} vs {label_2}', fontsize=16, fontweight='bold')
    
    # 1. Total simulation time comparison
    systems = [label_1, label_2]
    total_times = [total_1, total_2]
    bars1 = ax1.bar(systems, total_times, color=['skyblue', 'lightcoral'])
    ax1.set_title('Total Simulation Time')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, total_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average operation times comparison
    operations = ['PowerFlow', 'Snapshot', 'Iteration']
    times_1 = [avg_pf_1, avg_snap_1, avg_iter_1]
    times_2 = [avg_pf_2, avg_snap_2, avg_iter_2]
    
    x = np.arange(len(operations))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, times_1, width, label=label_1, color='skyblue')
    bars2b = ax2.bar(x + width/2, times_2, width, label=label_2, color='lightcoral')
    
    ax2.set_title('Average Operation Times')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(operations)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Power flow time distribution
    pf_times_1 = np.array(time_data_1.get('solve_powerflow_times', [0])) * 1000
    pf_times_2 = np.array(time_data_2.get('solve_powerflow_times', [0])) * 1000
    
    ax3.hist(pf_times_1, bins=20, alpha=0.7, label=label_1, color='skyblue', density=True)
    ax3.hist(pf_times_2, bins=20, alpha=0.7, label=label_2, color='lightcoral', density=True)
    ax3.set_title('Power Flow Time Distribution')
    ax3.set_xlabel('Time (milliseconds)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Iteration time series
    iter_times_1 = np.array(time_data_1.get('iteration_times', [0])) * 1000
    iter_times_2 = np.array(time_data_2.get('iteration_times', [0])) * 1000
    
    ax4.plot(range(len(iter_times_1)), iter_times_1, 'o-', label=label_1, color='skyblue', markersize=4)
    ax4.plot(range(len(iter_times_2)), iter_times_2, 's-', label=label_2, color='lightcoral', markersize=4)
    ax4.set_title('Iteration Times Over Simulation')
    ax4.set_xlabel('Iteration Number')
    ax4.set_ylabel('Time (milliseconds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary insights
    print(f"\n🔍 KEY INSIGHTS:")
    if pct_total > 5:
        print(f"   • {label_2} takes {pct_total:.1f}% longer overall than {label_1}")
    elif pct_total < -5:
        print(f"   • {label_2} is {abs(pct_total):.1f}% faster overall than {label_1}")
    else:
        print(f"   • Overall simulation times are similar (within 5%)")
    
    if abs(pct_pf) > 5:
        change_dir = "slower" if pct_pf > 0 else "faster"
        print(f"   • {label_2} power flow is {abs(pct_pf):.1f}% {change_dir} than {label_1}")
    
    if abs(pct_snap) > 5:
        change_dir = "slower" if pct_snap > 0 else "faster"
        print(f"   • {label_2} snapshots are {abs(pct_snap):.1f}% {change_dir} than {label_1}")
    
    return {
        'total_time_diff': diff_total,
        'total_time_pct': pct_total,
        'powerflow_diff': diff_pf,
        'powerflow_pct': pct_pf,
        'snapshot_diff': diff_snap,
        'snapshot_pct': pct_snap,
        'iteration_diff': diff_iter,
        'iteration_pct': pct_iter
    }

def save_performance_data(data, filename):
    """Save performance data to JSON file"""
    os.makedirs('./performance_data', exist_ok=True)
    filepath = f'./performance_data/{filename}.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")

def load_performance_data(filename):
    """Load performance data from JSON file"""
    filepath = f'./performance_data/{filename}.json'
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


#### PyPSA - IEEE 14 Bus System benchmark (no WEC)
ieee14_pypsa = wecgrid.Engine()
ieee14_pypsa.case("../grid/IEEE_14_bus.raw")
ieee14_pypsa.load(["pypsa"])
print("=== PyPSA: IEEE 14 (No WEC) ===")
ieee14_pypsa.simulate(num_steps=10)
ieee14_pypsa_time = ieee14_pypsa.pypsa.get_timing_data()
save_performance_data(ieee14_pypsa_time, "ieee14_pypsa_time")

# #### PyPSA - IEEE 14 Bus System benchmark (WEC)
ieee14_pypsa_wec = wecgrid.Engine()
ieee14_pypsa_wec.case("../grid/IEEE_14_bus.raw")
ieee14_pypsa_wec.load(["pypsa"])
ieee14_pypsa_wec.apply_wec(
   farm_name = "WEC-Farm",
   size = 1, # one RM3 in WEC farm  
   wec_sim_id = 1, # RM3 run id  
   bus_location=15, # create a new bus for farm  
   connecting_bus = 1, # Connect to bus 1 or swing bus
   scaling_factor = 1 # scale up the lab scale to about a 1kW
)
print("=== PyPSA: IEEE 14 (WEC) ===")
ieee14_pypsa_wec.simulate(num_steps=10)
ieee14_pypsa_wec_time = ieee14_pypsa_wec.pypsa.get_timing_data()
save_performance_data(ieee14_pypsa_wec_time, "ieee14_pypsa_wec_time")

############################################################################

#### PSS/E - IEEE 14 Bus System benchmark (no WEC)
ieee14_psse = wecgrid.Engine()
ieee14_psse.case("../grid/IEEE_14_bus.raw")
ieee14_psse.load(["psse"])
print("=== PSS/E: IEEE 14 (No WEC) ===")
ieee14_psse.simulate(num_steps=10)
ieee14_psse_time = ieee14_psse.psse.get_timing_data()
save_performance_data(ieee14_psse_time, "ieee14_psse_time")

# #### PSS/E - IEEE 14 Bus System benchmark (WEC)
ieee14_psse_wec = wecgrid.Engine()
ieee14_psse_wec.case("../grid/IEEE_14_bus.raw")
ieee14_psse_wec.load(["psse"])
ieee14_psse_wec.apply_wec(
   farm_name = "WEC-Farm",
   size = 1, # one RM3 in WEC farm  
   wec_sim_id = 1, # RM3 run id  
   bus_location=15, # create a new bus for farm  
   connecting_bus = 1, # Connect to bus 1 or swing bus
   scaling_factor = 1 # scale up the lab scale to about a 1kW
)
print("=== PSS/E: IEEE 14 (WEC) ===")
ieee14_psse_wec.simulate(num_steps=10)
ieee14_psse_wec_time = ieee14_psse_wec.psse.get_timing_data()
save_performance_data(ieee14_psse_wec_time, "ieee14_psse_wec_time")

#####################################################################
print("=== Plots / Metrics ===")
comparison_results_pypsa_ieee14 = compare_timing_data(
    ieee14_pypsa_time, 
    ieee14_pypsa_wec_time, 
    "PyPSA: IEEE 14 (No WEC)", 
    "PyPSA: IEEE 14 (With WEC)"
)

comparison_results_psse_ieee14 = compare_timing_data(
    ieee14_psse_time, 
    ieee14_psse_wec_time, 
    "PSS®E: IEEE 14 (No WEC)", 
    "PSS®E: IEEE 14 (With WEC)"
)

comparison_results_both_ieee14 = compare_timing_data(
    ieee14_pypsa_time, 
    ieee14_psse_time, 
    "PyPSA: IEEE 14 (No WEC)", 
    "PSS®E: IEEE 14 (No WEC)"
)

comparison_results_both_wec_ieee14 = compare_timing_data(
    ieee14_pypsa_wec_time, 
    ieee14_psse_wec_time, 
    "PyPSA: IEEE 14 (With WEC)", 
    "PSS®E: IEEE 14 (With WEC)"
)

# Save comparison results
save_performance_data(comparison_results_pypsa_ieee14, "comparison_pypsa_ieee14")
save_performance_data(comparison_results_psse_ieee14, "comparison_psse_ieee14")
save_performance_data(comparison_results_both_ieee14, "comparison_both_ieee14")
save_performance_data(comparison_results_both_wec_ieee14, "comparison_both_wec_ieee14")

