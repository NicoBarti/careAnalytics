import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data
from products.jam.readingGroup import compute_jaccard, compute_vicinity

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 5,
    'state_variables': ['H', 'N', 'T', 'SimpleB', 'Performance', 'MaxExp', 'SimpleE'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'dominance_lines.csv',
    'lambda_to_run': 4.0,
    'subDir': 'long300',
    'varsigma': 300,
    'fixed_kappa': 0.1
}

def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['varsigma'] = settings['varsigma']
    params['fixed_kappa'] = settings['fixed_kappa']
    state_vars = state_vars_override if state_vars_override else settings['state_variables']

    raw_data = generate_error_data(
        params=params, 
        reps=settings['reps'], 
        state_vars=state_vars, 
        initial_seed=params['seeds'], 
        work_dir=settings['base_working_dir'], 
        engine_path=settings['engine_path'], 
        treatment=f'{selection_name}/{settings["subDir"]}'
    )
    return raw_data

def main():
    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    # 2. Run Simulations and Collect Data
    mech_data = {}
    selection = 'dominance_lines'
    schedules = SETTINGS['treatments']
    chosen_lambda = SETTINGS['lambda_to_run']

    print(f"--- Calculating Data for Lambda: {chosen_lambda} ---")
    for schedule in schedules:
        subset = data.loc[(data['fixed_lambda'] == chosen_lambda) & (data['Pi'] == schedule)]
        if subset.empty:
            continue
        
        params = subset.iloc[0].copy()
        params['obsPerformance'] = True
        
        print(f"Running simulation for schedule: {schedule}...")
        try:
            mech_data[schedule] = process_simulation(params, SETTINGS, selection)
        except Exception as e:
            print(f"Error during simulation for schedule '{schedule}': {e}")

    if not mech_data:
        print("No data calculated. Exiting.")
        return

    # 3. Setup Global Figures
    fig_diagon, ax_diagon = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    
    # Setup for Heatmaps
    fig_heat, ax_heat = plt.subplot_mosaic(mosaic="""NRBc""", layout="constrained", figsize=(22, 7),
                                           width_ratios=[1, 1, 1, 0.05])
    
    col = SETTINGS['cmap']
    colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
    labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}

    # Collect Vicinity Data for all schedules
    all_vicinity_means = {}
    for schedule in schedules:
        if schedule in mech_data:
            data_dict = mech_data[schedule]
            needVicinity_all_reps = compute_vicinity(data_dict, 'N', "treated", all_replicates=True)
            all_vicinity_means[schedule] = needVicinity_all_reps.mean(0)

    # 4. Iterate over row_index and create plots
    #row_indices = np.arange(0, 301, 10)
    row_indices = np.array([5,60,170])
    i=0
    fig_grid, ax_grid = plt.subplots(nrows=1, ncols=3, figsize=(18, 7))
    for row_index in row_indices:
        fig_row, ax_row = plt.subplots(figsize=(12, 7))
        fig_zoom, ax_zoom = plt.subplots(figsize=(12, 7))

        
        for schedule in schedules:
            if schedule not in all_vicinity_means:
                continue
            
            color = colors[schedule]
            label = labels[schedule]
            needVicinity_mean = all_vicinity_means[schedule]

            # Check if row_index is within bounds for this matrix
            if row_index < needVicinity_mean.shape[0]:
                xs = np.arange(needVicinity_mean.shape[1])
                row_slice = needVicinity_mean[row_index, :]
                
                # Full Plot
                ax_row.plot(xs, row_slice, label=label, color=color, linewidth=2)
                
                # Zoomed Plot (+/- 20 cycles)
                start = max(0, row_index - 5)
                end = min(len(row_slice), row_index + 6)
                ax_zoom.plot(xs[start:end], row_slice[start:end], label=label, color=color, linewidth=2, marker='o', markersize=4)

                ax_grid[i].plot(xs[start:end], row_slice[start:end], label=label, color=color, linewidth=2, marker='o', markersize=4)

        ax_grid[i].axvline(x=row_index, color='gray', linestyle='--', alpha=0.5, label='Reference Cycle')
        ax_grid[i].set_title(f'Patients Receiving Treatment at Cycle {row_index}', fontsize=14)
        ax_grid[i].set_xlabel("Simulation Cycle", fontsize=14)
        ax_grid[i].set_ylabel("Average Needs", fontsize=14)
        ax_grid[i].legend(loc='upper right', fontsize=12)
        ax_grid[i].grid(True, linestyle=':', alpha=0.6)
        ax_grid[i].set_ylim(0, 10)
        i+=1


        # Finalize Full Plot
        ax_row.set_title(f"Average Needs of Treated Patients (Observation Cycle: {row_index})", fontsize=16)
        ax_row.set_xlabel("Simulation Cycle", fontsize=14)
        ax_row.set_ylabel("Average Needs", fontsize=14)
        ax_row.axvline(x=row_index, color='gray', linestyle='--', alpha=0.5, label='Treatment Cycle')
        ax_row.legend(loc='upper right', fontsize=12)
        ax_row.grid(True, linestyle=':', alpha=0.6)
        ax_row.set_ylim(0, 10)
        fig_row.tight_layout()
        fig_row.show()

        # Finalize Zoomed Plot
        ax_zoom.set_title(f"Average Needs Zoomed +/- 20 Cycles (Observation Cycle: {row_index})", fontsize=16)
        ax_zoom.set_xlabel("Simulation Cycle", fontsize=14)
        ax_zoom.set_ylabel("Average Needs", fontsize=14)
        ax_zoom.axvline(x=row_index, color='gray', linestyle='--', alpha=0.5, label='Treatment Cycle')
        ax_zoom.legend(loc='upper right', fontsize=12)
        ax_zoom.grid(True, linestyle=':', alpha=0.6)
        ax_zoom.set_ylim(0, 10)
        fig_zoom.tight_layout()
        fig_zoom.show()

    #Finalize grid plot
    fig_grid.suptitle('Progression of Needs for Patients that Received Treatment During a Cycle', fontsize=16)
    fig_grid.tight_layout()

    # 5. Global Plots (Diagonal Decay and Heatmaps)
    for schedule in schedules:
        if schedule not in mech_data:
            continue
        
        color = colors[schedule]
        label = labels[schedule]
        data_dict = mech_data[schedule]
        needVicinity_mean = all_vicinity_means[schedule]

        # Jaccard Analysis
        seekJac = np.array(compute_jaccard(data_dict, 'SimpleB', all_replicates=True))
        treatJac = np.array(compute_jaccard(data_dict, 'T', all_replicates=True))

        plot_temporal_series(ax=ax_diagon[0], data=seekJac.diagonal(offset=1, axis1=1, axis2=2), 
                             color=color, label=f'{label}', linestyle='solid', 
                             title='Seeking Behaviour Similarity', y_label='Jaccard Similarity Index')
        
        plot_temporal_series(ax=ax_diagon[1], data=treatJac.diagonal(offset=1, axis1=1, axis2=2), 
                             color=color, label=f'{label}', linestyle='solid', 
                             title='Treatment Delivery Similarity', y_label='Jaccard Similarity Index')

        # Heatmap
        if schedule == 'need':
            a = ax_heat['N']
        elif schedule == 'risk':
            a = ax_heat['R']
        else:
            a = ax_heat['B']
        sns.heatmap(needVicinity_mean, ax=a, vmin=0, vmax=10, cbar_ax=ax_heat['c'])
        a.set_title(label, fontsize=16)

    # Finalize Global Figures
    ax_diagon[0].set_title("Seeking Behaviour Similarity (Next Cycle Overlap)", fontsize=16)
    ax_diagon[0].set_xlabel("Time (Cycles)", fontsize=14)
    ax_diagon[0].legend(loc='upper left', fontsize=12)
    ax_diagon[1].set_title("Treatment Delivery Similarity (Next Cycle Overlap)", fontsize=16)
    ax_diagon[1].set_xlabel("Time (Cycles)", fontsize=14)
    ax_diagon[1].legend(loc='upper left', fontsize=12)
    fig_diagon.tight_layout()

    ax_heat['N'].set_ylabel("Time of Observation (Cycles)", fontsize=14)
    for k in ['N', 'R', 'B']:
        ax_heat[k].set_xlabel("Simulation Cycle", fontsize=14)
    fig_heat.suptitle("Average Needs of Treated Patients Over Time", fontsize=18)

    plt.show()

if __name__ == "__main__":
    main()
