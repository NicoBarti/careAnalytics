import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data, plot_histogram
from products.jam.readingGroup import compute_jaccard, compute_vicinity

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/finalH/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 5,
    'state_variables': ['H'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 300,
    'varsigma': 300,
    'csv_filename': 'dominance_lines.csv',
    'fixed_kappa': 0.1,
    
    # SENSITIVITY ANALYSIS CONFIGURATION
    # You can change 'param' to any variable recognized by the engine (e.g., 'varsigma', 'fixed_kappa', 'fixed_capN', 'fixed_lambda')
    'param' : 'fixed_lambda',
    'xlabel': 'Doctors\' Learning Rate',
    'values': np.linspace(start=0, stop=10, num=20),
    #'values': np.arange(100,300)

}


def process_simulation(params, settings, selection_name, param_name, param_value, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    # Set default values from SETTINGS
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['varsigma'] = settings['varsigma']
    params['fixed_kappa'] = settings['fixed_kappa']
    
    # Overwrite the specific parameter we are testing
    params[param_name] = param_value

    state_vars = state_vars_override if state_vars_override else settings['state_variables']

    # Use a unique sub-directory for this parameter/value combination to avoid cache collisions
    safe_value_str = str(param_value).replace('.', '_')
    treatment_path = f"{selection_name}/{param_name}_{safe_value_str}"

    raw_data = generate_error_data(
        params=params,
        reps=settings['reps'],
        state_vars=state_vars,
        initial_seed=params['seeds'],
        work_dir=settings['base_working_dir'],
        engine_path=settings['engine_path'],
        treatment=treatment_path
    )
    return raw_data

def main():
    # 1. Load Configuration Data
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

    results = {s: [] for s in SETTINGS['treatments']}
    errors = {s: [] for s in SETTINGS['treatments']}

    param_to_test = SETTINGS['param']
    test_values = SETTINGS['values']

    # 2. Run Simulations across parameter values
    for value in test_values:
        print(f"\n--- Testing {param_to_test} = {value} ---")
        
        for schedule in SETTINGS['treatments']:
            subset = data.loc[(data['Pi'] == schedule)]
            if subset.empty:
                continue

            params = subset.iloc[0].copy()
            
            try:
                raw_data = process_simulation(
                    params=params, 
                    settings=SETTINGS, 
                    selection_name='sensitivity_analysis',
                    param_name=param_to_test, 
                    param_value=value
                )
                
                # Extract H at last cycle for all replicates
                final_Hs = []
                for rep_data in raw_data.values():
                    # Get sum H of all patients at the last cycle
                    final_Hs.append(rep_data['H'].iloc[:, -1].sum())
                
                mean_final_h = np.mean(final_Hs)
                quantiles_fiinal_h = np.quantile(final_Hs, q=[0.05, 0.95])
                #Checking that A is 170 regardless of W
                # if value == 1 or value == 18:
                #     print(8)
                #     (np.array([raw_data[seed]['T'] for seed in raw_data.keys()])[:, :, 1] > 0).sum(1)

                #std_final_h = np.std(final_Hs)
                
                results[schedule].append(mean_final_h)
                errors[schedule].append(quantiles_fiinal_h)
                
                print(f"  {schedule}: mean H = {mean_final_h:.4f}")
                
            except Exception as e:
                print(f"  Error during simulation for {schedule} at value {value}: {e}")
                results[schedule].append(np.nan)
                errors[schedule].append(np.nan)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    col = SETTINGS['cmap']
    colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
    labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}

    for schedule in SETTINGS['treatments']:
        y = np.array(results[schedule])
        err = np.array(errors[schedule])
        x = np.array(test_values)

        # Filter out nans
        #mask = ~np.isnan(y)

        #if np.any(mask):
        #   ax.plot(x[mask], y[mask], 'o-', color=colors[schedule], label=labels[schedule], linewidth=2)

        ax.plot(x, y, '-', color=colors[schedule], label=labels[schedule])
        ax.errorbar(x, y, yerr=np.abs(err.T - y), fmt=',', ecolor=colors[schedule], elinewidth=0.2)



    #ax.set_title(f"Sensitivity Analysis: Final Disease Progression vs {SETTINGS['param_name']}", fontsize=16)
    ax.set_xlabel(SETTINGS['xlabel'], fontsize=14)
    ax.set_ylabel(f"Total Disease Progression in the Whole Population", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
