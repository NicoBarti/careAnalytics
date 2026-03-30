import pandas as pd
from products.jam.readingGroup import generate_error_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Import updated functions from refactored readingGroup.py
from products.jam.readingGroup import plot_temporal_series, generate_error_data, compute_jaccard, compute_vicinity, \
    plot_jaccard_analysis, plot_histogram
from products.jam.dominance_lines import get_average_N_treated, get_mean_trajectory, previous_encounters, corrExp_Health

# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance_lines/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 50,
    'state_variables': ['T', 'N'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'dominance_lines.csv',

    # Feature Flags
    'plot_lambda_series': False,  # Set to False to skip the main lambda comparison plot
    'add_risk_baseline': False,  # Set to False to skip adding the risk baseline line
    'plot_mechanism': True,  # Set to False to skip the detailed mechanism analysis
    'run_jaccard': False,  # Set to False to skip Jaccard analysis in mechanism section
    
    # Strategy Selection for Plotting
    'plot_quantile_schedules': ['need', 'risk', 'basal'] # Choose from ['need', 'basal', 'risk']
}

def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    state_vars = state_vars_override if state_vars_override else settings['state_variables']

    raw_data = generate_error_data(
        params=params, 
        reps=settings['reps'], 
        state_vars=state_vars, 
        initial_seed=params['seeds'], 
        work_dir=settings['base_working_dir'], 
        engine_path=settings['engine_path'], 
        treatment=f'{selection_name}/need'
    )
    return raw_data


def get_average_Exp_bound(raw_data, bound: float, lower = True):
    """Calculates the average expectation for patients whos needs are specified by bound"""

    #get the needs and exp
    needs_arr = np.array([run['N'] for run in raw_data.values()])
    simpleEs_arr = np.array([np.array(d['SimpleE'], dtype=float) for d in raw_data.values()])

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
        mech_vars = ['H','N', 'T', 'MaxExp', 'SimpleE']
        
        print(f"Running simulation for schedule: {schedule}...")
        try:
            mech_data = process_simulation(params, settings, selection, state_vars_override=mech_vars)
            all_mech_data[schedule] = mech_data
        except Exception as e:
            print(f"Error during simulation for schedule '{schedule}': {e}")

    return all_mech_data

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



    bounds = [.33]
    for bound in bounds:
        # --- Plotting Quantiles of Needs for Treated Patients (T > 0) ---
        #fig, ax = plt.subplots(figsize=(12, 9))
        #fig_q25, ax_q25 = plt.subplots(figsize=(12, 7))  # New figure for .25 quantile

        # Plotting expectations for Need quantiles at the population level
        fig_exp, ax_exp = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        col = mpl.colormaps['plasma']
        colors = {'need': col(0), 'risk': col(0.5), 'basal': col(0.9)}
        labels = {'need': 'Need-Prioritization', 'risk': 'Risk-Stratification', 'basal': 'FCFS'}

        for schedule in SETTINGS['plot_quantile_schedules']:
            if schedule not in mech_data:
                continue

            dT_dict = mech_data[schedule]
            #needs_list = [run['N'] for run in dT_dict.values()]
            #treats_list = [run['T'] for run in dT_dict.values()]

            #needs_arr = np.array(needs_list)
            #treats_arr = np.array(treats_list)

            #treated_needs = np.where(treats_arr == 0, np.nan, needs_arr)

            # Calculate 25th and 75th percentiles
            #quantiles = np.nanquantile(treated_needs, q=[bound, 1-bound], axis=1)
            #mean_quantiles = np.nanmean(quantiles, axis=1)

            #time_axis = np.arange(mean_quantiles.shape[1])
            #q25 = mean_quantiles[0, :]
            #q75 = mean_quantiles[1, :]

            #lower bound for plot 2
            #lowerBoundNeeds = np.nanquantile(treated_needs, q=bound, axis=1)

            #exps for plot 3
            exp_upper25 = get_average_Exp_bound(dT_dict, bound , lower=False)
            exp_lower25 = get_average_Exp_bound(dT_dict, bound, lower=True)


            # Plot 1: IQR Area Plot
            color = colors[schedule]
            #ax.plot(time_axis, q25, color=color, linestyle='--', linewidth=1, alpha=0.7)
            #ax.plot(time_axis, q75, color=color, linestyle='--', linewidth=1, alpha=0.7)
            #ax.fill_between(time_axis, q25, q75, color=color, alpha=0.3, label=labels[schedule])

            # Plot the mean of treated needs
            #mean_treated_needs = np.nanmean(np.nanmean(treated_needs, axis=1), axis=0)
            #ax.plot(time_axis, mean_treated_needs, color=color, linewidth=2)

            # Plot 2: Only .25 Quantile (for all three if in plot_quantile_schedules)
            #ax_q25.plot(time_axis, q25, color=color, linestyle='solid', linewidth=1)
            #plot_temporal_series(ax = ax_q25, data = lowerBoundNeeds, label=f'{labels[schedule]} ({bound} Needs Quantile)', color=color,
            #                     linestyle='solid', title=f'Needs for the Lower Bound of Needs for Treated Patients',
            #                     y_label=f'Needs at {bound} Quantile')

            #Plot 3: The expectations for the lower bound
            #ax_exp25.plot(time_axis, exp25, color=color, linestyle='solid', linewidth=1)
            plot_temporal_series(ax=ax_exp[0], data = exp_lower25, label=f'{labels[schedule]} ({bound} lower need quantile)', color=color,
                                 linestyle='solid', title=f'Lower Needs Patients',
                                 y_label=f'Mean Expectations for Patients Below the {bound} Needs Quantile')
            plot_temporal_series(ax=ax_exp[1], data = exp_upper25, label=f'{labels[schedule]} ({round(1-bound,2)} upper need quantile)', color=color,
                                 linestyle='solid', title=f'Higher Needs Patients',
                                 y_label=f'Mean Expectations Over the {round(1-bound,2)} Needs Quantile')

        # Finalize Figure 1
        #fig.suptitle('Needs at the Moment of Treatment Delivery', fontsize=16)
        #ax.set_xlabel('Simulation Time (Cycles)', fontsize=14)
        #ax.set_ylabel('Mean and IQR of Patient Needs at Appointment', fontsize=14)
        #ax.legend(loc='upper left')
        #ax.grid(True, linestyle=':', alpha=0.6)

        # Finalize Figure 2
        #ax_q25.set_title('Lower Need Bound for Accessing Care', fontsize=16)
        #ax_q25.set_xlabel('Simulation Time (Cycles)', fontsize=14)
        #ax_q25.set_ylabel(f'{bound} Needs Quantile in Patients who Received Treatment', fontsize=14)
        #ax_q25.legend(loc='upper left')
        #ax_q25.grid(True, linestyle=':', alpha=0.6)

        #Finalize Figure 3
        fig_exp.suptitle('Average Expectations for Patients Lower / Higer Needs (Population Level)', fontsize=16)
        ax_exp[0].set_xlabel('Simulation Time (Cycles)', fontsize=14)
        ax_exp[0].set_ylabel(f'Mean Expectations Below the {bound} Needs Quantile', fontsize=14)
        ax_exp[1].set_xlabel('Simulation Time (Cycles)', fontsize=14)
        ax_exp[1].set_ylabel(f'Mean Expectations Over the {round(1-bound,2)} Needs Quantile', fontsize=14)
        ax_exp[0].legend(loc='upper left')
        ax_exp[1].legend(loc='lower right')

        plt.tight_layout()
        #fig.show()
        #fig_q25.show()
        fig_exp.show()


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
