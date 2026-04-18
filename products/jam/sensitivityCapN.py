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
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/capH',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 10,
    'state_variables': ['H', 'T'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'capN_lines.csv',
    'lambda_to_run': 4.0,
    'bound': 1000 / 3500,  # = 1000 patients for N = 3500, so its the top or bottom 1000
    'fixed_kappa': 0.1,  # change accordingly to subDir

    'fixed_capN' : 'p', #change inside main [5,10,15]
    'subDir': 'p',  #

    'varsigma': 400,
    'jaccards': True,
    'capNs' : [5,10,15]
}


def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['varsigma'] = settings['varsigma']
    params['fixed_kappa'] = settings['fixed_kappa']
    params['fixed_capN'] = settings['fixed_capN']

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


def get_average_Exp_bound(raw_data, bound: float, lower=True):
    """Calculates the average expectation for patients whose needs are specified by bound"""
    # get the needs and exp
    needs_arr = np.array([run['N'] for run in raw_data.values()])
    simpleEs_arr = np.array([np.array(d['MaxExp'], dtype=float) for d in raw_data.values()])

    BoundExp = np.quantile(needs_arr, bound, axis=1)
    meanExp = []
    if lower:
        for i in range(BoundExp.shape[0]):
            meanExp.append(
                np.nanmean(np.where(needs_arr[i, :, :] <= BoundExp[i, :], simpleEs_arr[i, :, :], np.nan), axis=0))
    else:
        for i in range(BoundExp.shape[0]):
            meanExp.append(
                np.nanmean(np.where(needs_arr[i, :, :] >= BoundExp[i, :], simpleEs_arr[i, :, :], np.nan), axis=0))

    return np.array(meanExp)[:, 1:]


def care_seeking(raw_data):
    """Calculates the average number of attempts to seek care."""
    data_list = [np.array(d['SimpleB'], dtype=float) for d in raw_data.values()]
    return np.mean(np.array(data_list), axis=1)


def corrExp_Health(error_data, expectationType: str):
    """Calculates the correlation between Simple Epectations and Health. """

    simpleEs = np.array([np.array(d[expectationType], dtype=float) for d in error_data.values()])
    healths = np.array([np.array(d['H'], dtype=float) for d in error_data.values()])

    corrs = []
    for e in range(healths.shape[0]):
        # Per simulation
        sim_n, sim_u = simpleEs[e], healths[e]

        # Correlations per time window
        c_list = [np.corrcoef(sim_n[:, i], sim_u[:, i])[0, 1] for i in range(sim_n.shape[1])]
        corrs.append(c_list)

    return np.array(corrs)


def main():
    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    # Grid for the 3 caps
    fig_diagon, ax_diagon = plt.subplots(nrows=1, ncols=3, figsize=(18, 7))
    fig_performance, ax_performance = plt.subplots(nrows=1, ncols=3, figsize=(18,7))
    #axe
    i = 0
    for cap in SETTINGS['capNs']:
        # 2. Run Simulations and Collect Data
        mech_data = {}
        selection = 'dominance_lines'
        schedules = SETTINGS['treatments']
        chosen_lambda = SETTINGS['lambda_to_run']
        #Set the capN:
        SETTINGS['subDir']  = f'long{SETTINGS['varsigma']}_capN{cap}'
        SETTINGS['fixed_capN'] = cap

        print(f"--- Calculating Data for Lambda: {chosen_lambda} ---")
        for schedule in schedules:
            subset = data.loc[(data['fixed_lambda'] == chosen_lambda) & (data['Pi'] == schedule)]
            if subset.empty:
                print(f"Warning: No configuration found for schedule '{schedule}' with lambda {chosen_lambda}.")
                continue

            params = subset.iloc[0].copy()

            print(f"Running simulation for schedule: {schedule}...")
            try:
                mech_data[schedule] = process_simulation(params, SETTINGS, selection)
            except Exception as e:
                print(f"Error during simulation for schedule '{schedule}': {e}")

        if not mech_data:
            print("No data calculated. Exiting.")
            return


        col = SETTINGS['cmap']
        colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
        labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}

        print("\n--- Disease Progression Summary (at end of simulation) ---")
        for schedule in schedules:
            if schedule not in mech_data:
                continue

            color = colors[schedule]
            label = labels[schedule]
            data_dict = mech_data[schedule]


            # Fallback to T if Performance is not available
            treat_data = np.array(
                    [np.nanmean(np.where(d['T'] > 0, d['T'], np.nan), axis=0) for d in data_dict.values()])

            treatJac = np.array(compute_jaccard(data_dict, 'T', all_replicates=True))
            plot_temporal_series(ax=ax_diagon[i], data=treatJac.diagonal(offset=1, axis1=1, axis2=2), color=color,
                                 label=f'{label}, next cycle similarity', linestyle='solid', title='', y_label='')
            plot_temporal_series(ax=ax_performance[i], data=treat_data, color = color, label = f'{label}',
                                 linestyle='solid', title='', y_label='')
        ax_diagon[i].set_title(f"Needs Capped at {cap}")
        ax_diagon[i].set_xlabel("Time (Cycles)", fontsize=14)
        ax_diagon[i].set_ylabel("Treatment Similarity for Next Cycle", fontsize=14)
        ax_diagon[i].legend(loc='upper left', fontsize=12)
        ax_diagon[i].grid(True, linestyle=':', alpha=0.6)
        ax_diagon[i].set_ylim(0, 0.45)

        ax_performance[i].set_title(f"Needs Capped at {cap}")
        ax_performance[i].set_xlabel("Time (Cycles)", fontsize=14)
        ax_performance[i].set_ylabel("Average Needs Solved per Appointment", fontsize=14)
        ax_performance[i].legend(loc='upper right', fontsize=12)
        ax_performance[i].grid(True, linestyle=':', alpha=0.6)
        ax_performance[i].set_ylim(0, 2)


        i=i+1





    #fig_diagon.suptitle(f"Jaccard Similarity for Treatment Delivery (Next Cycle) for Different Caps", fontsize=16)
    fig_diagon.tight_layout()
    fig_diagon.show()
    fig_performance.tight_layout()
    fig_performance.show()

if __name__ == "__main__":
    main()
