import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data
from products.jam.dominance_lines import get_mean_trajectory

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/exploringPerformance/',
    'reps': 30,
    'state_variables': ['T', 'N', 'Performance'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'performance_lines.csv',
    'selection': 'exploring_performance',
    
    # --- MULTI-PARAMETER ANALYSIS CONFIGURATION ---
    # Define a dictionary of parameters and the list of values to test for each
    'sensitivity_configs': {
        'fixed_lambda': [0, 0.5, 1, 2, 4, 10],
        'totalCapacity': [100, 170, 200, 300],
        'fixed_psi': [0, 0.3, 0.6, 0.9, 1],
        'varsigma': [100, 300, 600]
    },
    
    # Human-readable labels for plotting
    'param_labels': {
        'fixed_lambda': 'Learning Rate ($\lambda$)',
        'totalCapacity': 'Total Capacity',
        'fixed_psi': 'Baseline Seeking ($\psi$)',
        'varsigma': 'Standard Deviation ($\varsigma$)'
    }
}

def process_simulation(params, settings, param_name, param_value):
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
        treatment=f"{settings['selection']}/{param_name}_{param_value}"
    )
    return raw_data

def main():
    # 1. Load Configuration Data (for baseline parameters)
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        data_path = f"/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/{SETTINGS['csv_filename']}"
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Could not find configuration file at {data_path}")
            return

    # Establish baseline configuration (from the first 'need' prioritization row)
    baseline_subset = data.loc[data['Pi'] == 'need']
    if baseline_subset.empty:
        baseline_params = data.iloc[0].copy()
    else:
        baseline_params = baseline_subset.iloc[0].copy()

    # 2. Iterate through each parameter configuration
    for target_param, test_values in SETTINGS['sensitivity_configs'].items():
        test_values = sorted(test_values)
        param_label = SETTINGS['param_labels'].get(target_param, target_param)
        
        print(f"\n--- Starting Analysis for {target_param} ---")
        
        # Setup Figure for this parameter
        fig, ax = plt.subplots(figsize=(20, 10), layout='constrained')
        norm = mpl.colors.Normalize(vmin=min(test_values), vmax=max(test_values))
        sm = plt.cm.ScalarMappable(cmap=SETTINGS['cmap'], norm=norm)
        sm.set_array([])

        for val in test_values:
            params = baseline_params.copy()
            color = SETTINGS['cmap'](norm(val))

            # Run Simulation
            try:
                print(f"  Simulating {target_param} = {val}...")
                raw_data = process_simulation(params, SETTINGS, target_param, val)
                
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
                print(f"  Simulation failed for {target_param}={val}: {e}")
                continue

        # Finalize Plot for this parameter
        ax.set_title(f"Impact of {param_label} on Need-Prioritization Efficiency", fontsize=30)
        ax.set_ylabel('Average needs treated by appointment', fontsize=25)
        ax.set_xlabel('Time (cycles)', fontsize=25)
        
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(param_label, fontsize=20)
        
        print(f"--- Finished Analysis for {target_param} ---")

    plt.show()

if __name__ == "__main__":
    main()
