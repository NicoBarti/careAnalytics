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
    'reps': 1,
    'state_variables': ['H', 'N', 'T', 'SimpleB', 'Performance', 'MaxExp', 'SimpleE'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'capH_lines.csv',
    'lambda_to_run': 4.0,
    'bound': 1000 / 3500,  # = 1000 patients for N = 3500, so its the top or bottom 1000

    'subDir': 'long300_capH5',  #
    'fixed_kappa': 0.1,  # change accordingly to subDir
    'fixed_capH' : 5,

    'varsigma': 300,
    'jaccards': True,
    'vecinity': False
}


def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['varsigma'] = settings['varsigma']
    params['fixed_kappa'] = settings['fixed_kappa']
    params['fixed_capH'] = settings['fixed_capH']

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
    fig_diagon, ax_diagon = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_focus, ax_focus = plt.subplots(figsize=(12, 7))

    # Grid Mosaic: Progression on left, Vertical Histograms on right
    axd = plt.figure(layout="constrained", figsize=(12, 7)).subplot_mosaic(
        """
        LB
        LR
        LN
        """,
        gridspec_kw=dict(width_ratios=[1.3, 1])
    )

    col = SETTINGS['cmap']
    colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
    labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}
    # Heatmap
    # fig_heat, ax_heat = plt.subplots(nrows=1, ncols=3, figsize=(22, 7))
    fig_heat, ax_heat = plt.subplot_mosaic(mosaic="""NRBc""", layout="constrained", figsize=(22, 7),
                                           width_ratios=[1, 1, 1, 0.07])

    bound = SETTINGS['bound']

    print("\n--- Disease Progression Summary (at end of simulation) ---")
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
            treat_data = np.array(
                [np.nanmean(np.where(d['T'] > 0, d['T'], np.nan), axis=0) for d in data_dict.values()])

        plot_temporal_series(ax=ax_delivery, data=treat_data, label=label, color=color, linestyle='solid',
                             title='Delivery of Treatments', y_label='Average Needs Solved per Appointment')
        ax_delivery.grid(True, linestyle=':', alpha=0.6)
        # 2. Progression of Diseases
        health_data = np.array([d['H'] for d in data_dict.values()]).mean(axis=1)
        plot_temporal_series(ax=ax_health, data=health_data, label=label, color=color, linestyle='solid',
                             title='Progression of Diseases', y_label='Average Health Problems per Patient')

        # Big Progression Plot on left of mosaic
        plot_temporal_series(ax=axd['L'], data=health_data, label=label, color=color, linestyle='solid',
                             title='Progression of Diseases', y_label='Average Health Problems per Patient')

        # 3. Care Seeking Behaviour
        seeking_data = care_seeking(data_dict)
        plot_temporal_series(ax=ax_seeking, data=seeking_data, label=label, color=color, linestyle='solid',
                             title='Care Seeking Behaviour', y_label='Proportion of the Population Seeking Care')

        # 4. Expectations and Correlations Combined Figure
        # Left: Average Expectations for Lower Needs Patients
        exp_lower = get_average_Exp_bound(data_dict, bound, lower=True)
        plot_temporal_series(ax=ax_exp_corr[1], data=exp_lower,
                             label=f'{label}, mean expectations in patients with lower need', color=color,
                             linestyle='solid', title='Expectations in Lower Needs Patients',
                             y_label=f'Mean Expectations in the lower {bound:.2f} Needs Quantile')

        # Right: Correlation Max. Expectations Health
        corrMaxExp = corrExp_Health(data_dict, expectationType='MaxExp')
        plot_temporal_series(ax=ax_exp_corr[0], data=np.array(corrMaxExp),
                             label=f'{label}, correlation between expectations and needs', color=color,
                             linestyle='solid',
                             title='Correlation between Expectations and Needs at the Population Level',
                             y_label='Pearson product-moment correlation of Needs and Expectations')

        # Focus:
        AllNeed = np.array([d['N'] for d in data_dict.values()])
        AllTreat = np.array([d['T'] for d in data_dict.values()])
        TreatmentNeed = np.where(AllTreat > 0, AllNeed, np.nan)
        q5, q25, q50, q75, q95 = np.nanquantile(TreatmentNeed, q=[0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
        xs = np.arange(q25.shape[1])
        alpha = 0.4 if schedule == 'need' else 0.2
        ax_focus.fill_between(xs, q25.mean(0), q75.mean(0), color=color, alpha=alpha,
                              label=f'{label}, Needs Median and IQR')
        # ax_focus.fill_between(xs,q5.mean(0), q95.mean(0), color=color, alpha=0.1)
        ax_focus.plot(xs, q25.mean(0), color=color, linestyle='--', linewidth=1, alpha=0.8)
        ax_focus.plot(xs, q75.mean(0), color=color, linestyle='--', linewidth=1, alpha=0.8)
        ax_focus.plot(xs, q50.mean(0), color=color, linestyle='solid', linewidth=1.5, alpha=1)
        ax_focus.plot(xs, q5.mean(0), color=color, linestyle='dotted', linewidth=1, alpha=0.8)
        ax_focus.plot(xs, q95.mean(0), color=color, linestyle='dotted', linewidth=1, alpha=0.8)

        # Jaccadrs
        if SETTINGS['jaccards']:
            seekJac = np.array(compute_jaccard(data_dict, 'SimpleB', all_replicates=True))
            treatJac = np.array(compute_jaccard(data_dict, 'T', all_replicates=True))
            plot_temporal_series(ax=ax_diagon[1], data=seekJac.diagonal(offset=1, axis1=1, axis2=2), color=color,
                                 label=f'{label}, next cycle similarity', linestyle='solid', title='', y_label='')
            plot_temporal_series(ax=ax_diagon[0], data=treatJac.diagonal(offset=1, axis1=1, axis2=2), color=color,
                                 label=f'{label}, next cycle similarity', linestyle='solid', title='', y_label='')

        # ax_diagon[0].plot(seekJac.diagonal(offset=1), color=color, label=f'{label}', linestyle='solid')
        # ax_diagon[1].plot(treatJac.diagonal(offset=1), color=color, label=f'{label}', linestyle='solid')

        # Vecinity
        if SETTINGS['vecinity']:
            # l = True if schedule == 'basal' else False
            if schedule == 'need':
                a = ax_heat['N']
            elif schedule == 'risk':
                a = ax_heat['R']
            else:
                a = ax_heat['B']
            needVecinity = compute_vicinity(data_dict, 'N', "treated", all_replicates=True)

            sns.heatmap(needVecinity.mean(0), ax=a, vmin=0, vmax=10, cbar_ax=ax_heat['c'])

        # 6. Histograms on right of mosaic (end of simulation)
        if schedule == 'basal':
            ax_idx = 'B'
        elif schedule == 'risk':
            ax_idx = 'R'
        elif schedule == 'need':
            ax_idx = 'N'

        plot_histogram(ax=axd[ax_idx], error_data=data_dict, title=f'{label} (End)',
                       color=color, xlab='Disease Progression', hist_label=label)

        # 7. Print Statistics to Console
        # Extract disease progression (H) at the last cycle for all replicates
        final_H = np.array([d['H'].iloc[:, -1] for d in data_dict.values()])  # (Reps, Patients)
        sums = final_H.sum(axis=1)  # (Reps,)
        mean_sum = np.mean(sums)
        ci_lower, ci_upper = np.quantile(sums, [0.025, 0.975])
        print(f"Strategy: {label}")
        print(f"  Sum of Disease Progression: {mean_sum:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")

    # Finalize Figures
    fig_delivery.tight_layout()
    fig_health.tight_layout()
    fig_seeking.tight_layout()
    ax_exp_corr[0].set_title('Population Level correlation between Expectations and Needs', fontsize=16)
    ax_exp_corr[0].set_xlabel('Time (Cycles)', fontsize=14)
    ax_exp_corr[0].set_ylabel('Correlation of Needs and Expectations', fontsize=14)
    ax_exp_corr[1].set_title('Expectations for the 1000 Lower-Needs Patients', fontsize=16)
    ax_exp_corr[1].set_xlabel('Time (Cycles)', fontsize=14)
    ax_exp_corr[1].set_ylabel('Mean Expectations for the lower 0.29 Needs Quantile', fontsize=14)
    ax_exp_corr[0].legend(loc='upper right', fontsize=12)
    ax_exp_corr[1].legend(loc='upper left', fontsize=12)
    ax_exp_corr[0].grid(True, linestyle=':', alpha=0.6)
    ax_exp_corr[1].grid(True, linestyle=':', alpha=0.6)

    ax_diagon[1].set_title("Seeking Behaviour - Next Cycle Similarity", fontsize=16)
    ax_diagon[1].set_xlabel("Time (Cycles)", fontsize=14)
    ax_diagon[1].set_ylabel("Jaccard Similarity for Next Cycle", fontsize=14)
    ax_diagon[1].legend(loc='upper left', fontsize=12)
    ax_diagon[0].set_title("Treatment Delivery - Next Cycle Similarity", fontsize=16)
    ax_diagon[0].set_xlabel("Time (Cycles)", fontsize=14)
    ax_diagon[0].set_ylabel("Jaccard Similarity for Next Cycle", fontsize=14)
    ax_diagon[0].legend(loc='upper left', fontsize=12)
    ax_diagon[0].grid(True, linestyle=':', alpha=0.6)
    ax_diagon[1].grid(True, linestyle=':', alpha=0.6)
    fig_diagon.tight_layout()
    fig_diagon.show()

    # Adding tick 1 to yaxis
    # yticks = ax_focus.get_yticks().tolist()
    # if 1 not in yticks:
    #    yticks.append(1)
    #    ax_focus.set_yticks(sorted(yticks))
    ax_focus.set_title('Needs at the Moment of Treatment', fontsize=16)
    ax_focus.set_xlabel('Time (Cycles)', fontsize=14)
    ax_focus.set_ylabel('Needs at the Moment of Treatment (Median, IQR, 5th, 95th)', fontsize=14)
    ax_focus.legend(loc='upper left', fontsize=12)
    fig_focus.tight_layout()

    ax_heat['N'].set_ylabel("Time (Cycles)", fontsize=14)
    ax_heat['N'].set_xlabel("Time (Cycles)", fontsize=14)
    ax_heat['N'].set_title("Need-Prioritisation", fontsize=16)
    ax_heat['R'].set_ylabel("Time (Cycles)", fontsize=14)
    ax_heat['R'].set_xlabel("Time (Cycles)", fontsize=14)
    ax_heat['R'].set_title("Risk-Stratification", fontsize=16)
    ax_heat['B'].set_ylabel("Time (Cycles)", fontsize=14)
    ax_heat['B'].set_xlabel("Time (Cycles)", fontsize=14)
    ax_heat['B'].set_title("FCFS", fontsize=16)
    ax_heat['c'].set_ylabel("Average Needs", fontsize=16)
    fig_heat.suptitle('Average Needs in Patients who Accessed Treatment (Jaccadard Matrices)', fontsize=16)
    fig_heat.show()

    # Uniform axes for the mosaic histograms
    maxY = []
    maxX = []
    for k in ['B', 'R', 'N']:
        maxY.append(axd[k].get_ylim()[1])
        maxX.append(axd[k].get_xlim()[1])
    for k in ['B', 'R', 'N']:
        axd[k].set_ylim(bottom=0, top=np.max(maxY))
        axd[k].set_xlim(left=0, right=np.max(maxX))

    axd['L'].legend(fontsize=12)
    axd['L'].set_title('Progression of Diseases', fontsize=16)
    axd['L'].set_xlabel('Average Progression per Patient', fontsize=14)
    axd['L'].set_ylabel('Time (Cycles)', fontsize=14)
    axd['L'].grid(True, linestyle=':', alpha=0.6)
    axd['B'].set_title('Progression at Cycle 300 - FCFS', fontsize=12)
    axd['B'].set_xlabel('Disease Progression', fontsize=10)
    axd['B'].set_ylabel('Patients', fontsize=10)
    axd['R'].set_title('Progression at Cycle 300 - Risk-Stratification', fontsize=12)
    axd['R'].set_xlabel('Disease Progression', fontsize=10)
    axd['R'].set_ylabel('Patients', fontsize=10)
    axd['N'].set_title('Progression at Cycle 300 - Need-Prioritisation', fontsize=12)
    axd['N'].set_xlabel('Disease Progression', fontsize=10)
    axd['N'].set_ylabel('Patients', fontsize=10)
    fig_delivery.tight_layout()
    fig_health.tight_layout()
    fig_seeking.tight_layout()
    # fig_heat.tight_layout()
    fig_exp_corr.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
