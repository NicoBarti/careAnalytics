import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 30,
    'state_variables': ['H', 'N', 'T', 'SimpleB', 'Performance', 'MaxExp', 'SimpleE'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'dominance_lines.csv',
    'lambda_to_run': 4.0,
    'bound': 0.33,
    'subDir': 'long300', #long300_zerokappa = 0, long300_lowkappa = 0.1, you need to uncomment params['fixed_kappa'] in proces_simulation()
    'varsigma': 300
}

def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['varsigma'] = settings['varsigma']
    #params['fixed_kappa'] = 0.1
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
            meanExp.append(np.nanmean(np.where(needs_arr[i, :, :] <= BoundExp[i, :], simpleEs_arr[i, :, :], np.nan), axis=0))
    else:
        for i in range(BoundExp.shape[0]):
            meanExp.append(np.nanmean(np.where(needs_arr[i, :, :] >= BoundExp[i, :], simpleEs_arr[i, :, :], np.nan), axis=0))

    return np.array(meanExp)[:, 1:]

def care_seeking(raw_data):
    """Calculates the average number of attempts to seek care."""
    data_list = [np.array(d['SimpleB'], dtype=float) for d in raw_data.values()]
    return np.mean(np.array(data_list), axis=1)

def corrExp_Health(error_data, expectationType:str):
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

    # 2. Run Simulations and Collect Data
    mech_data = {}
    selection = 'dominance_lines'
    schedules = SETTINGS['treatments']
    chosen_lambda = SETTINGS['lambda_to_run']

    print(f"--- Calculating Data for Lambda: {chosen_lambda} ---")
    for schedule in schedules:
        subset = data.loc[(data['fixed_lambda'] == chosen_lambda) & (data['Pi'] == schedule)]
        if subset.empty:
            print(f"Warning: No configuration found for schedule '{schedule}' with lambda {chosen_lambda}.")
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

    # 3. Setup Figures
    fig_delivery, ax_delivery = plt.subplots(figsize=(12, 7))
    fig_health, ax_health = plt.subplots(figsize=(12, 7))
    fig_seeking, ax_seeking = plt.subplots(figsize=(12, 7))
    # Unified figure for Expectations and Correlations
    fig_exp_corr, ax_exp_corr = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

    col = SETTINGS['cmap']
    colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
    labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}

    bound = SETTINGS['bound']

    for schedule in schedules:
        if schedule not in mech_data:
            continue
        
        color = colors[schedule]
        label = labels[schedule]
        data_dict = mech_data[schedule]

        # 1. Delivery of Treatments
        try:
            treat_data = np.array([d['Performance'] for d in data_dict.values()]).mean(axis=1)
            treat_data[treat_data == 0] = np.nan
        except KeyError:
            # Fallback to T if Performance is not available
            treat_data = np.array([np.nanmean(np.where(d['T'] > 0, d['T'], np.nan), axis=0) for d in data_dict.values()])

        plot_temporal_series(ax=ax_delivery, data=treat_data, label=label, color=color, linestyle='solid',
                             title='Delivery of Treatments', y_label='Average Needs Solved per Appointment')

        # 2. Progression of Diseases
        health_data = np.array([d['H'] for d in data_dict.values()]).mean(axis=1)
        plot_temporal_series(ax=ax_health, data=health_data, label=label, color=color, linestyle='solid',
                             title='Progression of Diseases', y_label='Average Health Problems per Patient')

        # 3. Care Seeking Behaviour
        seeking_data = care_seeking(data_dict)
        plot_temporal_series(ax=ax_seeking, data=seeking_data, label=label, color=color, linestyle='solid',
                             title='Care Seeking Behaviour', y_label='Proportion of the Population Seeking Care')

        # 4. Expectations and Correlations Combined Figure
        # Left: Average Expectations for Lower Needs Patients
        exp_lower = get_average_Exp_bound(data_dict, bound, lower=True)
        plot_temporal_series(ax=ax_exp_corr[1], data=exp_lower, label=f'{label}, mean expectations in patients with lower need', color=color,
                             linestyle='solid', title='Expectations in Lower Needs Patients',
                             y_label=f'Mean Expectations in the lower {bound:.2f} Needs Quantile')
        
        # Right: Correlation Max. Expectations Health
        corrMaxExp = corrExp_Health(data_dict, expectationType='MaxExp')
        plot_temporal_series(ax=ax_exp_corr[0], data=np.array(corrMaxExp), label=f'{label}, correlation between expectations and needs', color=color, linestyle='solid',
                             title='Correlation between Expectations and Needs at the Population Level', y_label='Pearson product-moment correlation of Needs and Expectations')

    # Finalize Figures
    fig_delivery.tight_layout()
    fig_health.tight_layout()
    fig_seeking.tight_layout()
    fig_exp_corr.suptitle(f'Analysis of Expectations', fontsize=16)
    fig_exp_corr.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
