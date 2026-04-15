import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data, compute_jaccard, compute_vicinity, \
    plot_jaccard_analysis, plot_histogram

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 5,
    'state_variables': ['T', 'N', 'SimpleB'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'dominance_lines.csv',
    'subDir': 'long300_lowkappa_01',
    'fixed_kappa': 0.1,  # change accordingly to subDir

    # Feature Flags
    'plot_lambda_series': False,   # Set to False to skip the main lambda comparison plot
    'add_risk_baseline': False,    # Set to False to skip adding the risk baseline line
    'plot_mechanism': True,       # Set to False to skip the detailed mechanism analysis
    'run_jaccard': True           # Set to False to skip Jaccard analysis in mechanism section
}

def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    # Ensure OBS_PERIOD is set
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['fixed_kappa'] = settings['fixed_kappa']

    
    state_vars = state_vars_override if state_vars_override else settings['state_variables']

    # Run Simulation
    raw_data = generate_error_data(
        params=params, 
        reps=settings['reps'], 
        state_vars=state_vars, 
        initial_seed=params['seeds'], 
        work_dir=settings['base_working_dir'], 
        engine_path=settings['engine_path'], 
        treatment=f'{selection_name}/{SETTINGS['subDir']}'
    )

    return raw_data

def get_mean_trajectory(raw_data, variable):
    """Extracts mean trajectory handling NaNs for a specific variable."""
    data_list = []
    for d in raw_data.values():
        arr = np.array(d[variable], dtype=float)
        arr[arr == 0] = np.nan
        data_list.append(arr)
    return np.array([np.nanmean(arr, axis=0) for arr in data_list])

def get_average_N_treated(raw_data, treated = True):
    """Calculates average N specifically for patients where T > 0."""
    data_list = []
    for d in raw_data.values():
        T = np.array(d['T'], dtype=float)
        N = np.array(d['N'], dtype=float)
        
        # Create mask where T > 0, or not treated (treated = False)
        if treated:
            treated_mask = T > 0
        else:
            treated_mask = (T == 0)

        
        # Get N values where T > 0
        # We calculate the mean N for treated patients at each time step
        # If no one is treated at a step, it will be NaN
        mean_N_treated = []
        for i in range(T.shape[1]):
            treated_in_step = treated_mask[:, i]
            if np.any(treated_in_step):
                mean_N_treated.append(np.mean(N[:, i][treated_in_step]))
            else:
                mean_N_treated.append(np.nan)
        
        data_list.append(np.array(mean_N_treated))
        
    return np.array(data_list)

def get_average_N_attempted(raw_data, allSought = True):
    """Calculates average N specifically for patients that attemptes and got or did not get an appointment."""
    data_list = []
    for d in raw_data.values():
        SimpleB = np.array(d['SimpleB'], dtype = float)
        T = np.array(d['T'], dtype=float)
        N = np.array(d['N'], dtype=float)

        #Create get mask:
        if allSought:
            sought = SimpleB > 0
        else:
            sought = np.where(T == 0, SimpleB, 0) > 0
        # Get N values where get > 0
        # We calculate the mean N for treated patients at each time step
        # If no one is treated at a step, it will be NaN
        mean_N_sought = []
        for i in range(T.shape[1]):
            sought_in_step = sought[:, i]
            if np.any(sought_in_step):
                mean_N_sought.append(np.mean(N[:, i][sought_in_step]))
            else:
                mean_N_sought.append(np.nan)

        data_list.append(np.array(mean_N_sought))

    return np.array(data_list)

def care_seeking(raw_data):
    """Calculates the average number of attempts to seek care."""
    data_list = [np.array(d['SimpleB'], dtype=float) for d in raw_data.values()]

    return np.mean(np.array(data_list), axis=1)

def simpleExpectations(raw_data,W):
    """Calculates the average number of expectations. As simple expetations are the sum of expetations accros providers,
    we need W to make it directly interpretable"""

    data_list = [np.array(d['SimpleE'], dtype=float)/W for d in raw_data.values()]
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



def previous_encounters(raw_data, ):
    """Calculates the average number of previous encounters."""
    data_list = []
    for d in raw_data.values():
        SimpleB = np.array(d['SimpleB'], dtype=float)
        T = np.array(d['T'], dtype=float)

        cumulativeInteractions = np.cumsum(T>0, axis=1)
        cumulativeOnlyTreated = np.where(T>0, cumulativeInteractions, np.nan) #only report on interaction moments T>0

        mean_previous = []
        for i in range(cumulativeOnlyTreated.shape[1]):
            previous_in_step = cumulativeOnlyTreated[:, i]
            if np.any(previous_in_step):
                mean_previous.append(np.nanmean(cumulativeOnlyTreated[:,i]))
            else:
                mean_previous.append(np.nan)

        data_list.append(np.array(mean_previous))

    return np.array(data_list)


def main():
    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    selection = 'dominance_lines'
    lambdas = sorted(data['fixed_lambda'].unique())

    # --- OPTIONAL: Main Lambda Series Plot ---
    if SETTINGS['plot_lambda_series']:
        fig, ax = plt.subplots(figsize=(20, 10), layout='constrained')
        
        norm = mpl.colors.Normalize(vmin=min(lambdas), vmax=max(lambdas))
        sm = plt.cm.ScalarMappable(cmap=SETTINGS['cmap'], norm=norm)
        sm.set_array([])

        print(f"Processing {len(lambdas)} lambda values for main plot...")
        for fix_lambda in lambdas:
            subset = data.loc[(data['fixed_lambda'] == fix_lambda) & (data['Pi'] == 'need')]
            if subset.empty: continue
            params = subset.iloc[0].copy()

            # Run Simulation
            try:
                raw_data = process_simulation(params, SETTINGS, selection)
                plot_data = get_mean_trajectory(raw_data, 'T')
            except Exception as e:
                print(f"Simulation failed for lambda={fix_lambda}: {e}")
                continue

            # Plot Line
            color = SETTINGS['cmap'](norm(fix_lambda))
            plot_temporal_series(
                ax=ax, 
                data=plot_data, 
                color=color, 
                label="",
                linestyle='solid', 
                title=None, 
                y_label='Average needs treated by appointment'
            )

        # --- OPTIONAL: Risk Routine Baseline ---
        if SETTINGS['add_risk_baseline']:
            max_lambda = 10.0
            risk_subset = data.loc[(data['fixed_lambda'] == max_lambda) & (data['Pi'] == 'risk')]
            if not risk_subset.empty:
                risk_params = risk_subset.iloc[0].copy()
                risk_params['OBS_PERIOD'] = SETTINGS['OBS_PERIOD']
                
                # Manually run simulation for risk
                risk_data = generate_error_data(
                    params=risk_params, 
                    reps=SETTINGS['reps'], 
                    state_vars=SETTINGS['state_variables'], 
                    initial_seed=risk_params['seeds'], 
                    work_dir=SETTINGS['base_working_dir'], 
                    engine_path=SETTINGS['engine_path'], 
                    treatment=f'{selection}/risk'
                )
                risk_plot_data = get_mean_trajectory(risk_data, 'T')
                
                plot_temporal_series(
                    ax=ax, 
                    data=risk_plot_data, 
                    color='black', 
                    label=f'Risk-Stratification Routine with $\\lambda = ${max_lambda}',
                    linestyle='dotted', 
                    title=None,
                    y_label='Average needs treated by appointment'
                )

        # Finalize Main Plot
        ax.set_title("Efficiency of the Need-Prioritization Routine for Different Learning Rates Across Time", fontsize=30)
        ax.set_ylabel('Average needs treated by appointment', fontsize=25)
        ax.set_xlabel('Time (cycles)', fontsize=25)
        ax.legend(fontsize=20)
        
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Learning Rate ($\\lambda$)', fontsize=20)
        fig.show()

    # --- OPTIONAL: Mechanism Analysis (Correlations & Jaccard) ---
    if SETTINGS['plot_mechanism']:
        # Note: Using hardcoded lambda=4.0 as per previous context logic, or specific logic
        # For simplicity, keeping the mechanism logic generic or for a specific lambda
        chosen_lambdas = [4.0]
        histFig, histAx = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), layout='constrained')
        needTreatFig, needTreatAx = plt.subplots(nrows=5, ncols=3, figsize=(20, 25), layout='constrained')
        axd = plt.figure(layout="constrained", figsize=(9, 12)).subplot_mosaic(
            """
            hhh
            147
            258
            369
            """,
            gridspec_kw=dict(width_ratios=[1, 1, 1], height_ratios=[1, 0.5, 0.5, 0.5])
        )
        # Aditional individual figures
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, layout="constrained")
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, layout="constrained")
        fig3, ax3 = plt.subplots(nrows=1, ncols=1, layout="constrained")
        fig4, ax4 = plt.subplots(nrows=1, ncols=1, layout="constrained")
        
        for chosen_lambda in chosen_lambdas:
            schedules = ['need', 'risk', 'basal']
            for schedule in schedules:
                subset = data.loc[(data['fixed_lambda'] == chosen_lambda) & (data['Pi'] == schedule)]
                if subset.empty: continue
                
                params = subset.iloc[0].copy()
                params['obsPerformance'] = True

                # Generate detailed data
                mech_vars = ['H','N', 'SimpleC', 'T', 'SimpleB', 'Performance', 'SimpleE', 'InstExp', 'MaxExp']
                mech_data = process_simulation(params, SETTINGS, selection, state_vars_override=mech_vars)


                # Plot histograms AX is the histogram figure, CA is the axe in the complex figure
                col = mpl.colormaps['plasma']
                if schedule == 'need':
                    colorHist = col(0)
                    AX = histAx[2]
                    acces_label = "Need-Prioritization Access Rule"
                    summary_label = "Need-Prioritization"
                    r = 7
                elif schedule == 'risk':
                    colorHist = col(0.5) # Approximate middle
                    AX = histAx[1]
                    acces_label = "Risk-Stratification Access Rule"
                    summary_label = "Risk-Stratification"
                    r = 8
                elif schedule == 'basal':
                    colorHist = col(0.9) # Approximate end
                    AX = histAx[0]
                    acces_label = "First-Come-First-Served Access Rule (FCFS)"
                    summary_label = "FCFS"
                    r= 9

                plot_histogram(ax=AX, error_data=mech_data, title=acces_label, color=colorHist, hist_label = acces_label)
                plot_histogram(ax = axd[str(r)], error_data=mech_data, title='Timestep 300', color=colorHist,
                               xlab=f'Disease Progression', hist_label = summary_label)
                axd[str(r)].legend()
                #plot the hist for the 200 timesteps
                partial_mech = {}
                for key in mech_data.keys():
                    partial_mech = {key: {'H': (mech_data[key]['H'].iloc)[:, 0:200]}}
                    #partial_mech[key]['H'] = (mech_data[key]['H'].iloc)[:, 0:200]
                plot_histogram(ax=axd[str(r-3)], error_data=partial_mech, title='Timestep 200', color=colorHist,
                               xlab=f'Disease Progression', hist_label = summary_label)
                axd[str(r-3)].legend()
                #plot the hist for the 100 timesteps
                partial_mech = {}
                for key in mech_data.keys():
                    partial_mech = {key: {'H': (mech_data[key]['H'].iloc)[:, 0:100]}}
                    #partial_mech[key]['H'] = (mech_data[key]['H'].iloc)[:, 0:100]
                plot_histogram(ax=axd[str(r-6)], error_data=partial_mech, title='Timestep 100', color=colorHist,
                               xlab=f'Disease Progression', hist_label = summary_label)
                axd[str(r-6)].legend()
                # Prepare Data Arrays
                need_data = np.array([d['N'] for d in mech_data.values()])
                use_data = np.array([d['SimpleC'] for d in mech_data.values()])
                
                # Treat data logic
                try:
                    treat_data = np.array([d['Performance'] for d in mech_data.values()]).mean(axis=1)
                    treat_data[treat_data == 0] = np.nan
                except KeyError:
                    # Fallback if 'Performance' variable issue
                    treat_data = get_mean_trajectory(mech_data, 'T')

                # Calculate Correlations & Percentages
                corrs, percents = [], []
                total_capacity = int(params['totalCapacity'])

                for e in range(need_data.shape[0]):
                    c_run, p_run = [], []
                    for i in range(need_data.shape[2]): # Time steps
                        # Correlation
                        c_run.append(np.corrcoef(need_data[e, :, i], use_data[e, :, i])[0, 1])
                        # Percentage
                        top_needs_idx = np.argpartition(-need_data[e, :, i], total_capacity)[:total_capacity]
                        p_run.append(use_data[e, :, i].take(top_needs_idx).sum() / total_capacity)
                    corrs.append(c_run)
                    percents.append(p_run)

                # Plot Mechanism Figure
                fig_mech, ax_mech = plt.subplots(nrows=1, ncols=2, constrained_layout=True, facecolor='white', figsize=(24, 8))
                ax_mech_twin_left = ax_mech[0].twinx()
                ax_mech_twin_right = ax_mech[1].twinx()


                # Left: Performance & Correlation
                plot_temporal_series(ax_mech[0], treat_data, "Performance", 'blue', 'solid', "Avg. Needs Solved per Visit", "Correlation between Needs and Use")
                plot_temporal_series(ax_mech_twin_left, np.array(corrs), "Correlation", 'green', 'dotted', "Correlation per cycle")
                
                # Right: Performance & Percentage
                plot_temporal_series(ax_mech[1], treat_data, "Performance", 'blue', 'solid', "Avg. Needs Solved per Visit", f"Percentage of the {total_capacity} more severe that got an appointment")
                plot_temporal_series(ax_mech_twin_right, np.array(percents), "Percentage", 'green', 'dotted', "Percentage per cycle")

                # Legends
                for ax, ax_twin in [(ax_mech[0], ax_mech_twin_left), (ax_mech[1], ax_mech_twin_right)]:
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax_twin.get_legend_handles_labels()
                    ax.legend(h1+h2, l1+l2, frameon=True)
                    ax.set_ylim(bottom=0)
                    ax_twin.legend().remove() # Remove duplicate legend on twin axis
                
                fig_mech.suptitle(f"Lambda: {chosen_lambda} for scheduling: {schedule}", fontsize = 15)
                fig_mech.show()

                # Jaccard Analysis
                if SETTINGS['run_jaccard']:
                    jaccard_results = {
                        'SimpleB': {schedule: compute_jaccard(mech_data, 'SimpleB')},
                        'T': {schedule: compute_jaccard(mech_data, 'T')}
                    }
                    seeker_results = {
                        'sN': {schedule: compute_vicinity(mech_data, 'N', "seekers")},
                        'tN': {schedule: compute_vicinity(mech_data, 'N', "treated")}
                    }
                    
                    # Call the plotting function with correct list
                    plot_jaccard_analysis(
                        jaccard_data=jaccard_results, 
                        seeker_data=seeker_results, 
                        treatments=[schedule],
                        optionalTitle=f' $\\lambda = ${chosen_lambda} - {schedule}'
                    )

                # Treatment-need metric:
                #for the treatment-need metric:
                av_N_T = get_average_N_treated(mech_data)
                plot_temporal_series(ax = needTreatAx[0][0], data = av_N_T, label=f'{schedule}', color=colorHist, linestyle='solid',
                                     title=f'Patients who received treatment', y_label='Average N')

                plot_temporal_series(ax = ax1, data = av_N_T, label=summary_label, color=colorHist, linestyle='solid',
                                     title=f'Needs at the Moment of Treatment Delivery', y_label='Average Patient Needs at the Appointment')
                av_N_T = get_average_N_treated(mech_data, treated=False)
                plot_temporal_series(ax=needTreatAx[0][2], data=av_N_T, label=f'{schedule}', color=colorHist,
                                     linestyle='solid',
                                     title=f'Patients did not receive treatment', y_label='Average N')
                av_N_attemptDidntGet = get_average_N_attempted(mech_data, allSought=False)
                plot_temporal_series(ax=needTreatAx[2][2], data=av_N_attemptDidntGet, label=f'{schedule}', color=colorHist,
                                     linestyle='solid',
                                     title=f'Attemptetd Not Treated', y_label='Average N')
                av_N_attemptGot = get_average_N_attempted(mech_data, allSought=True)
                plot_temporal_series(ax=needTreatAx[1][2], data=av_N_attemptGot, label=f'{schedule}', color=colorHist,
                                     linestyle='solid',
                                     title=f'Attempted', y_label='Average N')
                plot_temporal_series(ax = needTreatAx[0][1], data = np.array([mech_data[d]['N'] for d in mech_data.keys()]).mean(1), label=f'{schedule}', color=colorHist, linestyle='solid',
                                     title=f'All patients', y_label='Average N')
                plot_temporal_series(ax = needTreatAx[1][0], data = treat_data,  label=f'{schedule}', color=colorHist, linestyle='solid',
                                     title=f'Needs Solved per Appointment', y_label='Average N solver per appointment')
                plot_temporal_series(ax = ax2, data = treat_data, label=summary_label, color=colorHist, linestyle='solid',
                                     title=f'Delivery of Treatments', y_label='Average Needs Solved per Appointment')
                for a in [needTreatAx[1][1], axd['h']]:
                    plot_temporal_series(ax = a, data = np.array([mech_data[d]['H'] for d in mech_data.keys()]).mean(1),  label=acces_label, color=colorHist, linestyle='solid',
                                     title=f'Progression of Diseases', y_label='Average Health Problems per Patient')
                plot_temporal_series(ax = needTreatAx[2][0], data = np.array(corrs),  label=f'{schedule}', color=colorHist, linestyle='solid',
                                     title=f'Correlation', y_label='Correlation needs and access to treatment')
                plot_temporal_series(ax = needTreatAx[2][1], data = np.array(percents),  label=f'{schedule}', color=colorHist, linestyle='solid',
                                     title=f'% 170 More Severe Appointment', y_label='Percentaje')

                previous = previous_encounters(mech_data)
                plot_temporal_series(ax = needTreatAx[3][0], data = np.array(previous), label = summary_label, color = colorHist, linestyle='solid',
                                     title = 'Previous Appointments Treated', y_label='Average Number of Previous Appointments')

                seeking = care_seeking(mech_data)
                for aa in [needTreatAx[3][1], ax3]:
                    plot_temporal_series(ax=aa, data=np.array(seeking), label=summary_label,
                                     color=colorHist, linestyle='solid',
                                     title='Care seeking behaviour',
                                     y_label='Proportion of the Population Seeking Care')


                simexp = simpleExpectations(raw_data=mech_data, W = params['W'])
                plot_temporal_series(ax=needTreatAx[3][2], data=np.array(simexp), label = summary_label, color = colorHist, linestyle='solid',
                                     title = 'Evolution of Expectations During the Simulation', y_label='Average Expectations per Patient')

                corrSimpleExp = corrExp_Health(mech_data, expectationType='SimpleE')
                plot_temporal_series(ax=needTreatAx[4][0], data=np.array(corrSimpleExp), label = summary_label, color = colorHist, linestyle='solid',
                                     title = 'Correlation Simple Expectations Health', y_label='Correlation')
                corrInstExp = corrExp_Health(mech_data, expectationType='InstExp')
                plot_temporal_series(ax=needTreatAx[4][1], data=np.array(corrInstExp), label = summary_label, color = colorHist, linestyle='solid',
                                     title = 'Correlation Ins. Expectations Health', y_label='Correlation')
                corrMaxExp = corrExp_Health(mech_data, expectationType='MaxExp')
                plot_temporal_series(ax=needTreatAx[4][2], data=np.array(corrMaxExp), label = summary_label, color = colorHist, linestyle='solid',
                                     title = 'Correlation Max. Expectations Health', y_label='Correlation')





        histFig.show()
        needTreatFig.show()
        for column in [[1,2,3], [4,5,6], [7,8,9]]:
            maxX, maxY = [], []
            for r in column:
                maxX.append(axd[str(r)].get_xlim()[1])
                maxY.append(axd[str(r)].get_ylim()[1])
            for r in column:
                axd[str(r)].set_ylim(bottom=0, top=np.array(maxY).max())
                axd[str(r)].set_xlim(left=0, right=np.array(maxX).max())
        #
        # for r in range(1,10):
        #     maxX.append(axd[str(r)].get_xlim()[1])
        #     maxY.append(axd[str(r)].get_ylim()[1])
        # for r in range(1, 10):
        #     axd[str(r)].set_ylim(bottom=0, top = np.array(maxY).max())
        #     axd[str(r)].set_xlim(left=0, right = np.array(maxX).max())
        plt.show()


if __name__ == "__main__":
    main()
