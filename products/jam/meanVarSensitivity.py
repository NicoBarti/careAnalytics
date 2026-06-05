import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import generate_error_data

# --- SETTINGS ---
# Define the colormap once for consistency
col = mpl.colormaps['plasma']

SETTINGS = {
    'engine_path': '/Users/Nico/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/Nico/Desktop/simulationOutputs/Hetero_Disease_Exp/',
    'output_dir': '/Users/Nico/Desktop/simulationOutputs/JAMPaper/FinalNeedSensitivity',
    # this is for saving img to disk
    'reps': 10,
    'state_variables': ['H'],
    'OBS_PERIOD': 300,
    'csv_filename': 'dominance_lines.csv',
    'selection': 'long300_lowkappa_01',
    'total_patients': 3500,
    'subDir': 'long300_lowkappa_01',  # Added subDir for organizing outputs

    # Parameters to sweep and their values
    'sensitivity_configs': {
        'fixed_tau': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'fixed_lambda': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'totalCapacity': [0, 20, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310],
        'fixed_psi': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
        'W': [1, 11, 21, 31, 41, 45, 51, 61, 71],
        'fixed_rho': [0, 1, 2, 4, 6, 8, 10],
        'fixed_eta': [0, 1, 2, 4, 6, 8, 10],
        'fixed_kappa': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1],

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
    },
    # Updated grid positions for all 8 parameters
    'ax': {'totalCapacity': [0, 0],
           'fixed_tau': [0, 1],
           'fixed_lambda': [1, 0],
           'W': [1, 1],
           'fixed_rho': [2, 0],
           'fixed_eta': [2, 1],
           'fixed_psi': [3, 0],
           'fixed_kappa': [3, 1]},

    # Consistent colors for policies (from selected_plots.py)
    'policy_colors': {
        'need': col(0),  # Dark purple/blue
        'risk': col(0.5),  # Pink/Magenta
        'basal': col(0.9)  # Yellow
    },

    # Consistent labels for policies (from selected_plots.py)
    'policy_labels': {
        'need': 'Need-Prioritization',
        'risk': 'Risk-Stratification',
        'basal': 'FCFS'
    }
}


# Modified to return list of mean and std dev of final H per replicate
def process_simulation(params, settings, param_name, param_value, policy_name):
    """
    Runs the simulation and returns lists of mean final H and std dev of final H
    (across patients) for each replicate.
    """
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params[param_name] = param_value

    raw_data = generate_error_data(
        params=params,
        reps=settings['reps'],
        state_vars=settings['state_variables'],
        initial_seed=params['seeds'],
        work_dir=settings['base_working_dir'],
        engine_path=settings['engine_path'],
        treatment=f"{settings['selection']}/{policy_name}/{settings['subDir']}/{param_name}_{param_value}"
    )

    mean_final_H_per_replicate = []
    std_final_H_per_replicate = []
    for rep_data in raw_data.values():
        h_array = np.array(rep_data['H'], dtype=float)
        # h_array[:, 1] represents the final H values for all patients in this replicate
        mean_final_H_per_replicate.append(np.mean(h_array[:, 1]))
        std_final_H_per_replicate.append(np.std(h_array[:, 1]))

    return mean_final_H_per_replicate, std_final_H_per_replicate  # Return lists of values per replicate


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
            print(
                "Error: Neither '/Users/Nico' nor '/Users/nicolasbarticevic' found in base_working_dir. Please check paths.")
            return  # Exit if paths are completely unexpected

        print(f"Updated base working directory to: {SETTINGS['base_working_dir']}")
        print(f"Updated engine path to: {SETTINGS['engine_path']}")
        print(f"Updated output directory to: {SETTINGS['output_dir']}")

    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    # If the path was switched, and the original fallback was hardcoded, it needs to be adjusted too


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

    # Create the grid - Adjusted nrows to 4
    grid_fig, grid_ax = plt.subplots(nrows=4, ncols=2, figsize=(14, 16), layout='constrained')
    # Create twin axes for the grid subplots
    grid_ax2 = np.array([[ax_sub.twinx() for ax_sub in row] for row in grid_ax])

    # 2. Iterate through each parameter for a separate sensitivity plot
    for target_param, test_values in SETTINGS['sensitivity_configs'].items():
        test_values = sorted(test_values)
        param_label = SETTINGS['param_labels'].get(target_param, target_param)

        print(f"\n--- Sensitivity Analysis for {target_param} ---")
        fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
        ax2 = ax.twinx()  # Create twin axis for individual plot

        # 3. For each policy, calculate the "Final Health vs Parameter Value" line
        for policy in policies:
            print(f"  Processing Policy: {policy}")
            policy_subset = data.loc[data['Pi'] == policy]
            if policy_subset.empty: continue
            baseline_params = policy_subset.iloc[0].copy()

            # Identify the baseline value for this parameter
            baseline_val = baseline_params[target_param]

            # y_values will now be a list of lists (replicates for each test_value)
            mean_H_replicates_for_param_values = []
            std_H_replicates_for_param_values = []

            for val in test_values:
                print(f"    [{policy}] Val: {val}...", end='\r')
                # process_simulation now returns two lists of replicate results
                mean_H_reps, std_H_reps = process_simulation(baseline_params.copy(), SETTINGS, target_param, val,
                                                             policy)
                mean_H_replicates_for_param_values.append(mean_H_reps)
                std_H_replicates_for_param_values.append(std_H_reps)

            # Convert to numpy arrays for easier statistics (shape: [num_test_values, num_reps])
            mean_H_replicates_array = np.array(mean_H_replicates_for_param_values)
            std_H_replicates_array = np.array(std_H_replicates_for_param_values)

            # Calculate mean and quantiles across replicates for plotting
            mean_final_H_values = mean_H_replicates_array.mean(axis=1)
            mean_final_H_quantiles = np.quantile(mean_H_replicates_array, q=[0.05, 0.95], axis=1)

            mean_std_final_H_values = std_H_replicates_array.mean(axis=1)
            mean_std_final_H_quantiles = np.quantile(std_H_replicates_array, q=[0.05, 0.95], axis=1)

            color = SETTINGS['policy_colors'].get(policy, 'black')
            label = SETTINGS['policy_labels'].get(policy, policy.upper())

            # Plot Mean Final H on primary axis
            ax.plot(test_values, mean_final_H_values, marker='o', label=f'{label} (Mean H)', color=color, linewidth=2,
                    alpha=0.8)
            ax.errorbar(x=test_values, y=mean_final_H_values, yerr=np.abs(mean_final_H_values - mean_final_H_quantiles),
                        fmt='none', ecolor=color, capsize=5, alpha=0.3)

            # Plot Mean Std Dev Final H on twin axis
            ax2.plot(test_values, mean_std_final_H_values, marker='x', linestyle='--', label=f'{label} (Std Dev H)',
                     color=color, linewidth=1.5, alpha=0.6)
            ax2.errorbar(x=test_values, y=mean_std_final_H_values,
                         yerr=np.abs(mean_std_final_H_values - mean_std_final_H_quantiles), fmt='none', ecolor=color,
                         capsize=3, alpha=0.2)

            # Plot on the grid subplots
            c = SETTINGS['ax'].get(target_param, [0, 0])
            grid_ax[c[0], c[1]].plot(test_values, mean_final_H_values, marker='o', label=f'{label} (Mean H)',
                                     color=color, linewidth=2, alpha=0.8)
            grid_ax[c[0], c[1]].errorbar(x=test_values, y=mean_final_H_values,
                                         yerr=np.abs(mean_final_H_values - mean_final_H_quantiles), fmt='none',
                                         ecolor=color, capsize=5, alpha=0.3)
            grid_ax2[c[0], c[1]].plot(test_values, mean_std_final_H_values, marker='x', linestyle='--',
                                      label=f'{label} (Std Dev H)', color=color, linewidth=1.5, alpha=0.6)
            grid_ax2[c[0], c[1]].errorbar(x=test_values, y=mean_std_final_H_values,
                                          yerr=np.abs(mean_std_final_H_values - mean_std_final_H_quantiles), fmt='none',
                                          ecolor=color, capsize=3, alpha=0.2)

            # Special mark for the baseline value on both axes
            if baseline_val in test_values:
                idx = test_values.index(baseline_val)
                ax.plot(baseline_val, mean_final_H_values[idx], marker='*', markersize=15,
                        color=color, markeredgecolor='black', label='_nolegend_')
                ax2.plot(baseline_val, mean_std_final_H_values[idx], marker='*', markersize=15,
                         color=color, markeredgecolor='black', label='_nolegend_')
                grid_ax[c[0], c[1]].plot(baseline_val, mean_final_H_values[idx], marker='*', markersize=15,
                                         color=color, markeredgecolor='black', label='_nolegend_')
                grid_ax2[c[0], c[1]].plot(baseline_val, mean_std_final_H_values[idx], marker='*', markersize=15,
                                          color=color, markeredgecolor='black', label='_nolegend_')

        # Finalize Plot for individual figure
        ax.set_xlabel(param_label, fontsize=15)
        ax.set_ylabel('Average Final Health Status', fontsize=15, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, linestyle='--', alpha=0.7)

        ax2.set_ylabel('Std Dev Final Health Status', fontsize=15, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Combine legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        star_proxy = mpl.lines.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                                      markeredgecolor='black', markersize=12, label='Basal Scenario')
        ax.legend(handles=h1 + h2 + [star_proxy], loc='best', title="Scenario")

        print(f"\n--- Finished {target_param} ---")
        plt.show()

        # Finalize Plot for grid subplot
        grid_ax[c[0], c[1]].set_xlabel(param_label, fontsize=15)
        grid_ax[c[0], c[1]].set_ylabel('Average Final Health Status', fontsize=15, color='blue')
        grid_ax[c[0], c[1]].tick_params(axis='y', labelcolor='blue')
        grid_ax[c[0], c[1]].grid(True, linestyle='--', alpha=0.7)
        grid_ax[c[0], c[1]].set_ylim(bottom=0, top=30)  # Keep mean Y-limit consistent

        grid_ax2[c[0], c[1]].set_ylabel('Std Dev Final Health Status', fontsize=15, color='red')
        grid_ax2[c[0], c[1]].tick_params(axis='y', labelcolor='red')
        # You might want to set a consistent ylim for std dev across grid plots too
        # grid_ax2[c[0], c[1]].set_ylim(bottom=0, top=some_max_std_value) 

        # Combine legends for grid subplot
        h1_grid, l1_grid = grid_ax[c[0], c[1]].get_legend_handles_labels()
        h2_grid, l2_grid = grid_ax2[c[0], c[1]].get_legend_handles_labels()
        grid_ax[c[0], c[1]].legend(handles=h1_grid + h2_grid + [star_proxy], loc='best', title="Scenario", fontsize=8)

    grid_fig.suptitle('Health Impact Sensitivity Analysis (Mean & Std Dev)', fontsize=20)
    grid_fig.show()


if __name__ == "__main__":
    main()

