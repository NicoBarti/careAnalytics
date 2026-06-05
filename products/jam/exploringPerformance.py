import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data
from products.jam.dominance_lines import get_mean_trajectory

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/Nico/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/Nico/Desktop/simulationOutputs/JAMPaper/exploringPerformance/',
    'reps': 10,
    'state_variables': ['Performance'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'performance_lines.csv',
    'selection': 'exploring_performance',
    
    # --- MULTI-PARAMETER ANALYSIS CONFIGURATION ---
    'sensitivity_configs': {
        'fixed_lambda': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'totalCapacity': [110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310],
        'fixed_psi': [0, 0.5, 1],
        'W': [1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
        'fixed_rho': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'fixed_eta': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'fixed_kappa': [0, 0.2, 0.4, 0.6, 0.8, 1],
    },
    
    # Human-readable labels for plotting
    'param_labels': {
        'fixed_lambda': r'Learning Rate ($\lambda$)',
        'totalCapacity': r'Appointments per Cycle',
        'fixed_psi': r'Expectation-Driven Seeking ($\psi$)',
    }
}

def process_simulation(params, settings, param_name, param_value, policy_name):
    """Runs the simulation for a specific configuration."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    
    # Overwrite the specific target parameter
    params[param_name] = param_value
    
    raw_data = generate_error_data(
        params=params, 
        reps=settings['reps'], 
        state_vars=settings['state_variables'], 
        initial_seed=params['seeds'], 
        work_dir=settings['base_working_dir'], 
        engine_path=settings['engine_path'], 
        treatment=f"{settings['selection']}/{policy_name}/{param_name}_{param_value}"
    )
    return raw_data

def main():
    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    if not os.path.exists(data_path):
        # Handle path mismatch between machines
        data_path = data_path.replace('nicolasbarticevic', 'Nico')
        SETTINGS['base_working_dir'] = SETTINGS['base_working_dir'].replace('nicolasbarticevic', 'Nico')
        SETTINGS['engine_path'] = SETTINGS['engine_path'].replace('nicolasbarticevic', 'Nico')
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    # --- Get all unique policies from the 'Pi' column ---
    policies = data['Pi'].unique()
    print(f"Found policies to analyze: {policies}")

    # 2. Iterate through each Policy
    for policy in policies:
        print(f"\n==========================================")
        print(f"RUNNING ANALYSIS FOR POLICY: {policy}")
        print(f"==========================================")
        
        policy_subset = data.loc[data['Pi'] == policy]
        if policy_subset.empty:
            print(f"No data found for policy {policy}, skipping...")
            continue
        
        baseline_params = policy_subset.iloc[0].copy()

        # 3. Iterate through each parameter configuration
        for target_param, test_values in SETTINGS['sensitivity_configs'].items():
            test_values = sorted(test_values)
            param_label = SETTINGS['param_labels'].get(target_param, target_param)
            
            print(f"\n--- Starting Analysis for {target_param} (Policy: {policy}) ---")
            
            # Setup Figure
            fig, ax = plt.subplots(figsize=(20, 10), layout='constrained')
            norm = mpl.colors.Normalize(vmin=min(test_values), vmax=max(test_values))
            sm = plt.cm.ScalarMappable(cmap=SETTINGS['cmap'], norm=norm)
            sm.set_array([])

            for val in test_values:
                params = baseline_params.copy()
                color = SETTINGS['cmap'](norm(val))

                # Run Simulation
                try:
                    print(f"  [{policy}] Simulating {target_param} = {val}...")
                    raw_data = process_simulation(params, SETTINGS, target_param, val, policy)
                    
                    # Process data for plotting
                    try:
                        plot_data = np.array([d['Performance'] for d in raw_data.values()]).mean(axis=1)
                        plot_data[plot_data == 0] = np.nan
                    except (KeyError, ValueError):
                        plot_data = get_mean_trajectory(raw_data, 'T')

                    # Plot Line
                    plot_temporal_series(
                        ax=ax, 
                        data=plot_data, 
                        color=color, 
                        label="",
                        linestyle='solid', 
                        title=None, 
                        y_label='Average Needs Solved per Appointment'
                    )

                except Exception as e:
                    print(f"  Simulation failed for {policy}/{target_param}={val}: {e}")
                    continue

            # Finalize Plot
            ax.set_title(f"Policy: {policy.upper()} | Impact of {param_label}", fontsize=30)
            ax.set_ylabel('Average needs treated by appointment', fontsize=25)
            ax.set_xlabel('Time (cycles)', fontsize=25)
            
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(param_label, fontsize=20)
            
            print(f"--- Finished Analysis for {target_param} in {policy} ---")
            fig.show()

    plt.show()

if __name__ == "__main__":
    main()
