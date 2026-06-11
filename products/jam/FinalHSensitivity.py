import sys
import os
# Add the project root to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import generate_error_data

# --- SETTINGS ---
# Define the colormap once for consistency
col = mpl.colormaps['plasma']

BASE_WORKING_DIR = '/Users/Nico/Desktop/simulationOutputs/try/'

SETTINGS = {
    'engine_path': '/Users/Nico/Desktop/CareEngineAnalytics/engine/ABMServer7.jar',
    'base_working_dir': BASE_WORKING_DIR,
    'output_dir': BASE_WORKING_DIR, #this is for saving img to disk
    'reps': 1,
    'state_variables': ['H'],
    'OBS_PERIOD': 300,
    'csv_filename': 'dominance_lines.csv',
    'selection': 'sensitivity',
    'total_patients': 3500,
    'subDir': 'long300_lowkappa_01', # Added subDir for organizing outputs
    
    # Parameters to sweep and their values
    'sensitivity_configs': {
        'fixed_tau': [0,1,2,3,4,5,6,7,8,9,10],
        'fixed_lambda': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'totalCapacity': [0,20,30,50,70,90,110,130, 150, 170, 190, 210,230,250, 270,290, 310],
        'fixed_psi': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1], # Uncommented
        'W': [1, 11, 21, 31, 41, 45, 51, 61, 71],
        'fixed_rho': [0, 1,2, 4, 6, 8, 10],
        'fixed_eta': [0, 1,2, 4, 6, 8, 10],
        'fixed_kappa': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1], # Uncommented
        'prioritization_granularity': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], # Uncomment to sweep if Java engine supports it
    },
    
    # Human-readable labels for parameters (Raw strings r'' for LaTeX)
    'param_labels': {
        'fixed_lambda': r'Doctors\' Learning Rate ($\lambda$)',
        'totalCapacity': r'Total Appointments per Cycle (A)',
        'fixed_psi': r'Subjective Initiative ($\psi$)',
        'W': r'Number of Doctors (W) (constant appointments)',
        'fixed_rho': r'Positive Expectation Formation ($\rho$)',
        'fixed_eta': r'Negative Expectation Formation ($\eta$)',
        'fixed_kappa': r'Expectations Noise ($\kappa$)',
        'fixed_tau': r'Maximum Needs Treated per Appointment ($\tau$)',
        'prioritization_granularity': r'Prioritization Granularity ($g$)',
    },
    
    # Consistent colors for policies (from selected_plots.py)
    'policy_colors': {
        'need': col(0),     # Dark purple/blue
        'risk': col(0.5),   # Pink/Magenta
        'basal': col(0.9)   # Yellow
    },
    
    # Consistent labels for policies (from selected_plots.py)
    'policy_labels': {
        'need': 'Need-Prioritization',
        'risk': 'Risk-Stratification',
        'basal': 'FCFS'
    }
}

def process_simulation(params, settings, param_name, param_value, policy_name):
    """Runs the simulation and returns the total health problems normalized by patient count."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params[param_name] = param_value
    
    raw_data = generate_error_data(
        params=params, 
        reps=settings['reps'], 
        state_vars=settings['state_variables'], 
        initial_seed=params['seeds'], 
        work_dir=settings['base_working_dir'], 
        engine_path=settings['engine_path'], 
        treatment=f"{settings['selection']}/{policy_name}/{settings['subDir']}/{param_name}_{param_value}" # Updated treatment path
    )
    
    # Calculate Total Health Problems (Sum across all cycles for each replicate, divided by total patients)
    avg_health_problems = []
    var_health_problems = []
    for rep_data in raw_data.values():
        h_array = np.array(rep_data['H'], dtype=float)
        avg_health_problems.append(np.average(h_array, axis=0)[1])
        var_health_problems.append(np.std(h_array, axis=0)[1])

        
    return avg_health_problems, var_health_problems

def main():
    # --- Path Handling for multiple machines ---
    current_base_working_dir = SETTINGS['base_working_dir']
    current_engine_path = SETTINGS['engine_path']
    current_output_dir = SETTINGS['output_dir']

    # Check if the current base_working_dir exists
    if not os.path.exists(current_base_working_dir):
        print(f"Warning: Base working directory '{current_base_working_dir}' not found. Trying alternative path.")
        # Assume the other user's path
        if "/Users/Nico" in current_base_working_dir:
            SETTINGS['base_working_dir'] = current_base_working_dir.replace("/Users/Nico", "/Users/nicolasbarticevic")
            SETTINGS['engine_path'] = current_engine_path.replace("/Users/Nico", "/Users/nicolasbarticevic")
            SETTINGS['output_dir'] = current_output_dir.replace("/Users/Nico", "/Users/nicolasbarticevic")
        elif "/Users/nicolasbarticevic" in current_base_working_dir:
            SETTINGS['base_working_dir'] = current_base_working_dir.replace("/Users/nicolasbarticevic", "/Users/Nico")
            SETTINGS['engine_path'] = current_engine_path.replace("/Users/nicolasbarticevic", "/Users/Nico")
            SETTINGS['output_dir'] = current_output_dir.replace("/Users/nicolasbarticevic", "/Users/Nico")
        else:
            print("Error: Neither '/Users/Nico' nor '/Users/nicolasbarticevic' found in base_working_dir. Please check paths.")
            return # Exit if paths are completely unexpected

        print(f"Updated base working directory to: {SETTINGS['base_working_dir']}")
        print(f"Updated engine path to: {SETTINGS['engine_path']}")
        print(f"Updated output directory to: {SETTINGS['output_dir']}")

    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    # If the path was switched, and the original fallback was hardcoded, it needs to be adjusted too
    if not os.path.exists(data_path) and "/Users/nicolasbarticevic" in SETTINGS['base_working_dir']:
        data_path = f"/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/exploringPerformance/{SETTINGS['csv_filename']}"
    elif not os.path.exists(data_path): # Original fallback if no switch happened or switch failed
        data_path = f"/Users/Nico/Desktop/simulationOutputs/JAMPaper/exploringPerformance/{SETTINGS['csv_filename']}"


    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    # Ensure output directory exists
    if not os.path.exists(SETTINGS['output_dir']):
        os.makedirs(SETTINGS['output_dir'])
        print(f"Created output directory: {SETTINGS['output_dir']}")

    policies = data['Pi'].unique()


    active_params = list(SETTINGS['sensitivity_configs'].keys())
    num_params = len(active_params)
    ncols = 3
    nrows = int(np.ceil(num_params / ncols))

    # Create the two grids: one for mean H, one for standard deviation H
    grid_fig_mean, grid_ax_mean = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    grid_fig_std, grid_ax_std = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    
    # Flatten grid axes for easy 1D indexing
    axes_mean_flat = grid_ax_mean.flatten() if nrows > 1 else np.array([grid_ax_mean]).flatten()
    axes_std_flat = grid_ax_std.flatten() if nrows > 1 else np.array([grid_ax_std]).flatten()
    
    # Hide any unused subplots
    for i in range(num_params, len(axes_mean_flat)):
        axes_mean_flat[i].set_visible(False)
        axes_std_flat[i].set_visible(False)
    
    # 2. Iterate through each parameter for a separate sensitivity plot
    for target_param, test_values in SETTINGS['sensitivity_configs'].items():
        test_values = sorted(test_values)
        param_label = SETTINGS['param_labels'].get(target_param, target_param)
        param_idx = active_params.index(target_param)
        ax_mean_g = axes_mean_flat[param_idx]
        ax_std_g = axes_std_flat[param_idx]
        
        print(f"\n--- Sensitivity Analysis for {target_param} ---")
        fig_mean, ax_mean = plt.subplots(figsize=(12, 7), layout='constrained')
        fig_std, ax_std = plt.subplots(figsize=(12, 7), layout='constrained')

        # 3. For each policy, calculate the "Final Health vs Parameter Value" line
        for policy in policies:
            print(f"  Processing Policy: {policy}")
            policy_subset = data.loc[data['Pi'] == policy]
            if policy_subset.empty: continue
            baseline_params = policy_subset.iloc[0].copy()
            
            # Identify the baseline value for this parameter
            baseline_val = baseline_params.get(target_param, None)
            
            y_values = []
            yy_values = []
            for val in test_values:
                print(f"    [{policy}] Val: {val}...", end='\r')
                health_normalized = process_simulation(baseline_params.copy(), SETTINGS, target_param, val, policy)
                y_values.append(health_normalized[0])
                yy_values.append(health_normalized[1])

            
            # Process statistics for plotting
            y_array = np.array(y_values)
            mean_values = y_array.mean(axis=1)
            mean_quantiles = np.quantile(y_array, q=[0.05, 0.95], axis=1)
            yy_array = np.array(yy_values)
            var_values = yy_array.mean(axis=1)
            var_quantiles = np.quantile(yy_array, q=[0.05, 0.95], axis=1)
            
            color = SETTINGS['policy_colors'].get(policy, 'black')
            label = SETTINGS['policy_labels'].get(policy, policy.upper())
            
            # --- Mean H Plots ---
            # Individual plot
            ax_mean.plot(test_values, mean_values, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_mean.errorbar(x=test_values, y=mean_values, yerr=np.abs(mean_values - mean_quantiles), fmt='none', ecolor=color, capsize=5, alpha=0.3)
            # Grid plot
            ax_mean_g.plot(test_values, mean_values, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_mean_g.errorbar(x=test_values, y=mean_values, yerr=np.abs(mean_values - mean_quantiles), fmt='none', ecolor=color, capsize=5, alpha=0.3)

            # --- Std Dev H Plots ---
            # Individual plot
            ax_std.plot(test_values, var_values, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_std.errorbar(x=test_values, y=var_values, yerr=np.abs(var_values - var_quantiles), fmt='none', ecolor=color, capsize=5, alpha=0.3)
            # Grid plot
            ax_std_g.plot(test_values, var_values, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_std_g.errorbar(x=test_values, y=var_values, yerr=np.abs(var_values - var_quantiles), fmt='none', ecolor=color, capsize=5, alpha=0.3)

            # Special mark for the baseline value
            if baseline_val in test_values:
                idx = test_values.index(baseline_val)
                # Mean H baseline
                ax_mean.plot(baseline_val, mean_values[idx], marker='*', markersize=15, 
                            color=color, markeredgecolor='black', label='_nolegend_')
                ax_mean_g.plot(baseline_val, mean_values[idx], marker='*', markersize=15,
                            color=color, markeredgecolor='black', label='_nolegend_')
                # Std Dev H baseline
                ax_std.plot(baseline_val, var_values[idx], marker='*', markersize=15, 
                            color=color, markeredgecolor='black', label='_nolegend_')
                ax_std_g.plot(baseline_val, var_values[idx], marker='*', markersize=15,
                            color=color, markeredgecolor='black', label='_nolegend_')

        # Finalize Mean H Plot
        ax_mean.set_xlabel(param_label, fontsize=15)
        ax_mean.set_ylabel('Average Health Status', fontsize=15)
        ax_mean.grid(True, linestyle='--', alpha=0.7)
        # Grid Mean H:
        ax_mean_g.set_xlabel(param_label, fontsize=15)
        ax_mean_g.set_ylabel('Average Health Status', fontsize=15)
        ax_mean_g.grid(True, linestyle='--', alpha=0.7)
        ax_mean_g.set_ylim(bottom=0, top=30)
        
        # Mean Legends
        handles_mean, labels_mean = ax_mean.get_legend_handles_labels()
        star_proxy = mpl.lines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
                                      markeredgecolor='black', markersize=12, label='Basal Scenario')
        handles_mean.append(star_proxy)
        ax_mean.legend(handles=handles_mean)
        
        handles_g_mean, labels_g_mean = ax_mean_g.get_legend_handles_labels()
        handles_g_mean.append(star_proxy)
        ax_mean_g.legend(handles=handles_g_mean)

        # Finalize Std Dev H Plot
        ax_std.set_xlabel(param_label, fontsize=15)
        ax_std.set_ylabel('Standard Deviation of Health Status', fontsize=15)
        ax_std.grid(True, linestyle='--', alpha=0.7)
        # Grid Std Dev H:
        ax_std_g.set_xlabel(param_label, fontsize=15)
        ax_std_g.set_ylabel('Standard Deviation of Health Status', fontsize=15)
        ax_std_g.grid(True, linestyle='--', alpha=0.7)
        ax_std_g.set_ylim(bottom=0)
        
        # Std Dev Legends
        handles_std, labels_std = ax_std.get_legend_handles_labels()
        handles_std.append(star_proxy)
        ax_std.legend(handles=handles_std)
        
        handles_g_std, labels_g_std = ax_std_g.get_legend_handles_labels()
        handles_g_std.append(star_proxy)
        ax_std_g.legend(handles=handles_g_std)

        # Save the figures
        save_path_mean = os.path.join(SETTINGS['output_dir'], f"sensitivity_{target_param}_mean.png")
        fig_mean.savefig(save_path_mean, dpi=300)
        save_path_std = os.path.join(SETTINGS['output_dir'], f"sensitivity_{target_param}_std.png")
        fig_std.savefig(save_path_std, dpi=300)
        print(f"\n--- Finished {target_param}. Saved to {save_path_mean} and {save_path_std} ---")

        # Close the individual figures to prevent blocking during execution
        plt.close(fig_mean)
        plt.close(fig_std)

    last_subdir = os.path.basename(os.path.normpath(SETTINGS['base_working_dir']))
    grid_fig_mean.suptitle(f'Health Impact Sensitivity Analysis (Mean) - {last_subdir}', fontsize=20)
    
    # Save the grid figures
    grid_save_path_mean = os.path.join(SETTINGS['output_dir'], f"grid_sensitivity_mean_{last_subdir}.png")
    grid_fig_mean.savefig(grid_save_path_mean, dpi=300)
    print(f"Saved Mean H grid plot to {grid_save_path_mean}")
    
    grid_fig_std.suptitle(f'Health Impact Sensitivity Analysis (Standard Deviation) - {last_subdir}', fontsize=20)
    grid_save_path_std = os.path.join(SETTINGS['output_dir'], f"grid_sensitivity_std_{last_subdir}.png")
    grid_fig_std.savefig(grid_save_path_std, dpi=300)
    print(f"Saved Std Dev H grid plot to {grid_save_path_std}")

    # Call plt.show() once at the end of the script to display the grid plots
    plt.show()
if __name__ == "__main__":
    main()
