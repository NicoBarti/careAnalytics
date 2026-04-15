import pandas as pd
from products.jam.readingGroup import generate_error_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import  numpy.ma as ma

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data, compute_jaccard, compute_vicinity, \
    plot_jaccard_analysis, plot_histogram
from products.jam.dominance_lines import get_average_N_treated, get_mean_trajectory, previous_encounters, corrExp_Health

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 30,
    'state_variables': ['T', 'N', 'H', 'SimpleB'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'dominance_lines.csv',
    'varsigma': 300,
    'subDir': 'long300_lowkappa_01', #change the varsigma here too and pay attention to directory
    'fixed_kappa': 0.1, #change accordingly to subDir

    # Feature Flags
    'plot_lambda_series': False,  # Set to False to skip the main lambda comparison plot
    'add_risk_baseline': False,  # Set to False to skip adding the risk baseline line
    'plot_mechanism': True,  # Set to False to skip the detailed mechanism analysis
    'run_jaccard': True,  # Set to False to skip Jaccard analysis in mechanism section
    
    # Strategy Selection for Plotting
    'plot_quantile_schedules': ['need', 'risk', 'basal'] # Choose from ['need', 'basal', 'risk']
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


def get_average_Exp_bound(raw_data, bound: float, lower = True):
    """Calculates the average expectation for patients whos needs are specified by bound"""

    #get the needs and exp
    needs_arr = np.array([run['N'] for run in raw_data.values()])
    simpleEs_arr = np.array([np.array(d['MaxExp'], dtype=float) for d in raw_data.values()])

    # Extract the expect for the lower bound
    #if lower:
    #    BoundExp = np.where(needs_arr < np.quantile(needs_arr, bound, axis=0), simpleEs_arr, np.nan)
    #else:
    #    BoundExp = np.where(needs_arr > np.quantile(needs_arr, bound, axis=0), simpleEs_arr, np.nan)

    BoundExp = np.quantile(needs_arr, bound, axis=1)
    meanExp = []
    if lower:
        for i in range(BoundExp.shape[0]):
            meanExp.append(np.nanmean(np.where(needs_arr[i, :, :] <= BoundExp[i, :], simpleEs_arr[i, :, :], np.nan), axis = 0))
    else:
        for i in range(BoundExp.shape[0]):
            meanExp.append(np.nanmean(np.where(needs_arr[i, :, :] >= BoundExp[i, :], simpleEs_arr[i, :, :], np.nan), axis = 0))


    return np.array(meanExp)[:,1:]


def calculate_mechanism_data(data, settings, chosen_lambda=4.0, schedules=['need', 'risk', 'basal']):
    """
    Calculates and returns mechanism data for different schedules without plotting.

    Args:
        data (pd.DataFrame): The configuration data from the CSV.
        settings (dict): The main SETTINGS dictionary.
        chosen_lambda (float): The specific lambda value to analyze.
        schedules (list): A list of schedule names to process.

    Returns:
        dict: A dictionary where keys are schedule names and values are the raw simulation data.
    """
    all_mech_data = {}
    selection = 'dominance_lines'
    
    print(f"--- Calculating Mechanism Data for Lambda: {chosen_lambda} ---")
    for schedule in schedules:
        subset = data.loc[(data['fixed_lambda'] == chosen_lambda) & (data['Pi'] == schedule)]
        if subset.empty:
            print(f"Warning: No configuration found for schedule '{schedule}' with lambda {chosen_lambda}. Skipping.")
            continue
        
        params = subset.iloc[0].copy()
        params['obsPerformance'] = True

        # Define the state variables needed for mechanism analysis
        #mech_vars = ['H', 'N', 'SimpleC', 'T', 'SimpleB', 'Performance', 'MaxExp']
        mech_vars = ['H','N', 'T', 'MaxExp', 'SimpleB']
        
        print(f"Running simulation for schedule: {schedule}...")
        try:
            mech_data = process_simulation(params, settings, selection, state_vars_override=mech_vars)
            all_mech_data[schedule] = mech_data
        except Exception as e:
            print(f"Error during simulation for schedule '{schedule}': {e}")

    return all_mech_data

def subgroup_care_seeking(raw_data, quartileHealthCut, type, selectionVar):
    """Calculates the average number of attempts to seek care among a subgroup of Health Problems patients."""
    seek = np.array([np.array(d['SimpleB'], dtype=float) for d in raw_data.values()])
    health = np.array([np.array(d[selectionVar], dtype=float) for d in raw_data.values()])

    cut = np.quantile(health, quartileHealthCut, axis=1)
    selected_seek = []
    if type == 'lower':
        for i in range(cut.shape[0]):
            selected_seek.append(np.where(health[i, :] <= cut[i, :], seek[i, :, :], np.nan))
    elif type == 'upper':
        for i in range(cut.shape[0]):
            selected_seek.append(np.where(health[i, :] >= cut[i, :], seek[i, :, :], np.nan))
    else:
        raise ValueError("Invalid type. Choose 'lower' or 'upper'.")

    return np.nanmean(np.array(selected_seek), axis=1)

def subgroup_seeking_getting(raw_data, quartileHealthCut, type, selectionVar):
    """Calculates the average number of attempts to seek and how many of those got care among a subgroup of Health Problems patients."""
    seek = np.array([np.array(d['SimpleB'], dtype=float) for d in raw_data.values()])
    health = np.array([np.array(d[selectionVar], dtype=float) for d in raw_data.values()])
    got = np.array([np.array(d['T'], dtype=float) for d in raw_data.values()])
    got[got > 0] = 1 ## Trick for getting who access care, because T is always positive is accessed care.

    cut = np.quantile(health, quartileHealthCut, axis=1)
    selected_seek = []
    sought_got = []
    if type == 'lower':
        for i in range(cut.shape[0]):
            selected_seek.append(np.where(health[i, :] <= cut[i, :], seek[i, :, :], -1))
    elif type == 'upper':
        for i in range(cut.shape[0]):
            selected_seek.append(np.where(health[i, :] >= cut[i, :], seek[i, :, :], -1))

    else:
        raise ValueError("Invalid type. Choose 'lower' or 'upper'.")

    selected_seek = np.array(selected_seek)
    #Mask patients who didn't seek care in the got array:
    got = ma.array(got, mask=selected_seek < 1)
    #Mask patients outside the cut range: (marked as -1)
    selected_seek = ma.array(selected_seek, mask=selected_seek < 0)

    return ma.sum(selected_seek, axis=1), ma.sum(got, axis=1)

def main():
    """
    Runs a specific mechanism data calculation and plots quantiles of needs for treated patients.
    """
    # 1. Load Configuration Data
    data_path = f"{SETTINGS['base_working_dir']}/{SETTINGS['csv_filename']}"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    # 2. Define the specific parameters for the call
    lambda_to_run = 4.0
    schedules_to_process = list(set(SETTINGS['plot_quantile_schedules'] + ['need', 'risk','basal'])) # Ensure we have data for summary
    mech_data = {}
    
    # 3. Call the calculator function
    for schedule in schedules_to_process:
        res = calculate_mechanism_data(
            data=data,
            settings=SETTINGS,
            chosen_lambda=lambda_to_run,
            schedules=[schedule]
        )
        if res:
            mech_data[schedule] = res[schedule]

    if not mech_data:
        print("No data calculated. Exiting.")
        return



    #bounds = np.linspace(0, 1, 20) # Generate 20 points between 0 and 1
    bounds = [1000/3500] # = 1000 patients for N = 3500, so its the top or bottom 1000
    for bound in bounds:
        # --- Plotting Quantiles of Needs for Treated Patients (T > 0) ---
        fig, ax = plt.subplots(figsize=(12, 9))
        fig_q25, ax_q25 = plt.subplots(figsize=(12, 7))  # New figure for .25 quantile

        # Plotting expectations for Need quantiles at the population level
        fig_exp, ax_exp = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
        fig4, ax4 = plt.subplots(figsize=(12, 7))

        #Plotting the seeking behaviour by subgroups
        fig_seek, ax_seek = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
        fig_seek2, ax_seek2 = plt.subplots(nrows=1, ncols=2, figsize=(17,7))

        #Plotting those who sought and then got
        fig_seek_get1, ax_seek_get1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), layout = 'constrained')
        fig_seek_get2, ax_seek_get2 = plt.subplots(nrows=1, ncols=2, figsize=(17,7), layout = 'constrained')

        col = mpl.colormaps['plasma']
        colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
        labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}

        for schedule in SETTINGS['plot_quantile_schedules']:
            if schedule not in mech_data:
                continue

            dT_dict = mech_data[schedule]
            needs_list = [run['N'] for run in dT_dict.values()]
            treats_list = [run['T'] for run in dT_dict.values()]

            needs_arr = np.array(needs_list)
            treats_arr = np.array(treats_list)

            treated_needs = np.where(treats_arr == 0, np.nan, needs_arr)

            # Calculate 25th and 75th percentiles
            quantiles = np.nanquantile(treated_needs, q=[bound, 1-bound], axis=1)
            mean_quantiles = np.nanmean(quantiles, axis=1)

            time_axis = np.arange(mean_quantiles.shape[1])
            #q25 = mean_quantiles[0, :]
            #q75 = mean_quantiles[1, :]

            #lower bound for plot 2
            lowerBoundNeeds = np.nanquantile(treated_needs, q=bound, axis=1)

            #exps for plot 3
            exp_upper25 = get_average_Exp_bound(dT_dict, 1-bound , lower=False)
            exp_lower25 = get_average_Exp_bound(dT_dict, bound, lower=True)


            # Plot 1: IQR Area Plot
            color = colors[schedule]
            #q25 = mean_quantiles[0, :]
            #q75 = mean_quantiles[1, :]
            q25, q50, q75 = np.nanquantile(treated_needs, q=[bound, 0.5, 1-bound], axis=(1))
            q25, q50, q75 = np.mean(q25, axis=0), np.mean(q50, axis=0), np.mean(q75, axis=0)

            ax.plot(time_axis, q25, color=color, linestyle='--', linewidth=1, alpha=0.7)
            ax.plot(time_axis, q75, color=color, linestyle='--', linewidth=1, alpha=0.7)
            ax.fill_between(time_axis, q25, q75, color=color, alpha=0.3, label=labels[schedule])

            # Plot the mean of treated needs
            #mean_treated_needs = np.nanmean(np.nanmean(treated_needs, axis=1), axis=0)
            #median_treated_needs = np.nanmedian(np.nanmedian(treated_needs, axis=1), axis=0)
            ax.plot(time_axis, q50, color=color, linewidth=2)

            # Plot 2: Only .25 Quantile (for all three if in plot_quantile_schedules)
            ax_q25.plot(time_axis, q25, color=color, linestyle='solid', linewidth=1)
            plot_temporal_series(ax = ax_q25, data = lowerBoundNeeds, label=f'{labels[schedule]} ({bound:.2f} Needs Quantile)', color=color,
                                 linestyle='solid', title=f'Needs for the Lower Bound of Needs for Treated Patients (Bound: {bound:.2f})',
                                 y_label=f'Needs at {bound:.2f} Quantile')


            #Plot 3: The expectations for the lower bound
            #ax_exp25.plot(time_axis, exp25, color=color, linestyle='solid', linewidth=1)
            plot_temporal_series(ax=ax_exp[0], data = exp_lower25, label=f'{labels[schedule]} ({bound:.2f} lower need quantile)', color=color,
                                 linestyle='solid', title=f'Lower Need Patients',
                                 y_label=f'Mean Expectations for Patients Below the {bound:.2f} Needs Quantile')
            plot_temporal_series(ax=ax_exp[1], data = exp_upper25, label=f'{labels[schedule]} ({round(bound,2):.2f} upper need quantile)', color=color,
                                 linestyle='solid', title=f'Higher Needs Patients',
                                 y_label=f'Mean Expectations for Patients Over the {round(bound,2):.2f} Needs Quantile')

            expectations = np.array([d['MaxExp'] for d in mech_data[schedule].values()])
            plot_temporal_series(ax=ax4, data=np.mean(expectations,axis=1), label=labels[schedule], color=color,
                                 linestyle='solid',
                                 title='Expectations Evolution', y_label='Expectations')

            #Plot 4: The seeking behaviour according to sub-groups
            seeking = subgroup_care_seeking(dT_dict, bound, 'lower', selectionVar='N')
            plot_temporal_series(ax=ax_seek[0], data=seeking, label=labels[schedule], color=color,
                                 linestyle='solid',
                                 title='Seeking Behaviour Lower Needs', y_label='Seeking for Lower Needs Patients')
            seeking = subgroup_care_seeking(dT_dict, 1-bound, 'upper', selectionVar='N')
            plot_temporal_series(ax=ax_seek[1], data=seeking, label=labels[schedule], color=color,
                                 linestyle='solid',
                                 title='Seeking Behaviour Upper Needs', y_label='Seeking for Upper Needs Patients')
            seeking = subgroup_care_seeking(dT_dict, bound, 'lower', selectionVar='N')
            plot_temporal_series(ax=ax_seek2[0], data=seeking, label=labels[schedule], color=color,
                                 linestyle='solid',
                                 title='Seeking Behaviour Lower Health Problems', y_label='Seeking for Lower Health Problems Patients')
            seeking = subgroup_care_seeking(dT_dict, 1-bound, 'upper', selectionVar='N')
            plot_temporal_series(ax=ax_seek2[1], data=seeking, label=labels[schedule], color=color,
                                 linestyle='solid',
                                 title='Seeking Behaviour Upper Health Problems', y_label='Seeking for Upper Health Problems Patients')

            #Plot 5 and 6: seeking and getting
            var = 'H' #can use N, but the quantiles become unestable and you get lots of patients on the upper end
            #LOWER NEED
            seeking, getting = subgroup_seeking_getting(dT_dict, bound, 'lower', selectionVar=var)
            plot_temporal_series(ax=ax_seek_get1[0], data=seeking, label=f'1000 Patients with Lower Needs who Sought Care', color=color,
                                 linestyle='solid',
                                 title='Seeking Behaviour in the 1000 Patients with Lower Needs', y_label=f'Number of Patients who Sought Care')
            plot_temporal_series(ax=ax_seek_get1[1], data=getting, label=f'1000 Patients with Lower Needs who Got an Appointment',
                             color=color,
                             linestyle='solid',
                             title='Appointment Allocation in the 1000 Patients with Lower Needs',
                             y_label=f'Number of Patients that Got an Appointment')
            #HIGHER NEED
            seeking, getting = subgroup_seeking_getting(dT_dict, 1-bound, 'upper', selectionVar=var)
            plot_temporal_series(ax=ax_seek_get2[0], data=seeking, label=f'1000 Patients with Higher Needs who Sought Care', color=color,
                                 linestyle='solid',
                                 title='Seeking Behaviour in the 1000 Patients with Higher Needs', y_label=f'Number of Patients who Sought Care')
            plot_temporal_series(ax=ax_seek_get2[1], data=getting, label=f'1000 Patients with Higher Needs who Got an Appointment',
                             color=color,
                             linestyle='solid',
                             title='Appointment Allocation in the 1000 Patients with Higher Needs',
                             y_label=f'Number of Patients who Got an Appointment')

        # Finalize Figure 1
        fig.suptitle('Needs at the Moment of Treatment Delivery', fontsize=16)
        ax.set_xlabel('Simulation Time (Cycles)', fontsize=14)
        ax.set_ylabel('Median and IQR of Needs at the Moment of Treatment', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)

        # Finalize Figure 2
        ax_q25.set_title(f'Lower Need Bound for Accessing Care (Bound: {bound:.2f})', fontsize=16)
        ax_q25.set_xlabel('Simulation Time (Cycles)', fontsize=14)
        ax_q25.set_ylabel(f'{bound:.2f} Needs Quantile in Patients who Received Treatment', fontsize=14)
        ax_q25.legend(loc='upper left')
        ax_q25.grid(True, linestyle=':', alpha=0.6)

        #Finalize Figure 3
        fig_exp.suptitle(f'Average Expectations for Patients Lower / Higher Needs (Population Level, Bound: {bound:.2f})', fontsize=16)
        ax_exp[0].set_xlabel('Simulation Time (Cycles)', fontsize=14)
        ax_exp[0].set_ylabel(f'Mean Expectations Below the {bound:.2f} Needs Quantile', fontsize=14)
        ax_exp[1].set_xlabel('Simulation Time (Cycles)', fontsize=14)
        ax_exp[1].set_ylabel(f'Mean Expectations Over the {round(1-bound,2):.2f} Needs Quantile', fontsize=14)
        ax_exp[0].legend(loc='upper left')
        ax_exp[1].legend(loc='lower right')

        #Finalize Figure 5 and 6
        ax_seek_get2[0].legend(loc='lower right')
        ax_seek_get1[0].legend(loc='upper left')
        ax_seek_get1[1].legend(loc='upper left')


        plt.tight_layout()
        fig.show()
        fig_q25.show()
        fig_exp.show()
        fig_seek.show()
        fig_seek2.show()
        fig4.show()
        fig_seek_get1.show()
        fig_seek_get2.show()



    # (Keep summary prints)
    print("\n--- Summary Statistics ---")
    for schedule in schedules_to_process:
        if schedule in mech_data:
            TT = np.array([d['T'] for d in mech_data[schedule].values()]).sum(axis=(1,2))/3500
            H = np.array([d['H'] for d in mech_data[schedule].values()])
            HH = H.mean(axis=1)
            print(f"Schedule: {schedule}")
            print(f"  Total T per patient: mean={TT.mean():.4f}")
            print(f"  Final H average: mean={HH.mean(0)[-1]:.4f}")

if __name__ == "__main__":
    main()
