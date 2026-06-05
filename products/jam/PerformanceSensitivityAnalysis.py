
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
    'base_working_dir': '/Users/Nico/Desktop/simulationOutputs/JAMPaper/PerformanceSensitivity/',
    'reps': 30,
    'state_variables': ['Performance'],
    'OBS_PERIOD': 1,
    'csv_filename': 'performance_lines.csv',
    'selection': 'sensitivity_analysis',
    
    # Parameters to sweep and their values
    'sensitivity_configs': {
        'fixed_lambda': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'totalCapacity': [110, 150, 190, 230, 270, 310],
        'fixed_psi': [0, 0.5, 1],
        'W': [1, 11, 21, 31, 41],
        'fixed_rho': [0, 2, 4, 6, 8, 10],
        'fixed_eta': [0, 2, 4, 6, 8, 10],
        'fixed_kappa': [0, 0.2, 0.4, 0.6, 0.8, 1],
    },
    
    # Human-readable labels for parameters (Raw strings r'' for LaTeX)
    'param_labels': {
        'fixed_lambda': r'Learning Rate ($\lambda$)',
        'totalCapacity': r'Total Capacity',
        'fixed_psi': r'Baseline Seeking ($\psi$)',
        'W': r'Window Size (W)',
        'fixed_rho': r'Rho ($\rho$)',
        'fixed_eta': r'Eta ($\eta$)',
        'fixed_kappa': r'Kappa ($\kappa$)'
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
    """Runs the simulation and returns the sum of performance."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
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
    
    # Calculate Total Performance (Sum across all cycles, averaged across reps)
    total_performances = []
    for rep_data in raw_data.values():
        perf_array = np.array(rep_data['Performance'])
        total_performances.append(np.sum(perf_array))
        
    return np.mean(total_performances)

def main():
    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    if not os.path.exists(data_path):
        data_path = f"/Users/Nico/Desktop/simulationOutputs/JAMPaper/dominance_lines/{SETTINGS['csv_filename']}"
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    policies = data['Pi'].unique()
    
    # 2. Iterate through each parameter for a separate sensitivity plot
    for target_param, test_values in SETTINGS['sensitivity_configs'].items():
        test_values = sorted(test_values)
        param_label = SETTINGS['param_labels'].get(target_param, target_param)
        
        print(f"\n--- Sensitivity Analysis for {target_param} ---")
        fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
        
        # 3. For each policy, calculate the "Total Performance vs Parameter Value" line
        for policy in policies:
            print(f"  Processing Policy: {policy}")
            policy_subset = data.loc[data['Pi'] == policy]
            if policy_subset.empty: continue
            baseline_params = policy_subset.iloc[0].copy()
            
            y_values = []
            for val in test_values:
                print(f"    [{policy}] Val: {val}...", end='\r')
                total_perf = process_simulation(baseline_params.copy(), SETTINGS, target_param, val, policy)
                y_values.append(total_perf)
            
            # Plot the line for this policy using consistent colors and labels
            color = SETTINGS['policy_colors'].get(policy, 'black')
            label = SETTINGS['policy_labels'].get(policy, policy.upper())
            ax.plot(test_values, y_values, marker='o', label=label, color=color, linewidth=2)

        # Finalize Plot
        ax.set_title(f"Sensitivity Analysis: {param_label}", fontsize=20)
        ax.set_xlabel(param_label, fontsize=15)
        ax.set_ylabel('Total Sum of Performance (Entire Simulation)', fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Policy")
        
        print(f"\n--- Finished {target_param} ---")
        plt.show()

if __name__ == "__main__":
    main()
