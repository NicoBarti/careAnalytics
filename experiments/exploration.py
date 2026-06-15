import os
import sys
import argparse
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MyClasses.client import Client

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_output_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"exploration_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# --- ANALYSIS HELPERS ---

def get_average_Exp_bound(raw_data, bound: float, lower=True):
    needs_arr = np.array([np.array(run['N'], dtype=float) for run in raw_data.values()])
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
    data_list = [np.array(d['SimpleB'], dtype=float) for d in raw_data.values()]
    return np.mean(np.array(data_list), axis=1)

def corrExp_Health(error_data, expectationType: str):
    simpleEs = np.array([np.array(d[expectationType], dtype=float) for d in error_data.values()])
    healths = np.array([np.array(d['H'], dtype=float) for d in error_data.values()])

    corrs = []
    for e in range(healths.shape[0]):
        sim_n, sim_u = simpleEs[e], healths[e]
        c_list = [np.corrcoef(sim_n[:, i], sim_u[:, i])[0, 1] for i in range(sim_n.shape[1])]
        corrs.append(c_list)
    return np.array(corrs)

def compute_jaccard(data_dict, state_var):
    # Averaged over replicates
    all_matrices = []
    for simulation in data_dict.values():
        df_var = simulation[state_var]
        num_windows = df_var.shape[1]
        jaccard_matrix = np.empty((num_windows, num_windows))
        data = df_var.values > 0
        
        for i in range(num_windows):
            for j in range(num_windows):
                union = data[:, i] | data[:, j]
                intersection = data[:, i] & data[:, j]
                if not union.any():
                    jaccard_matrix[i, j] = np.nan
                else:
                    jaccard_matrix[i, j] = intersection.sum() / union.sum()
        all_matrices.append(jaccard_matrix)
    return np.array(all_matrices)

def compute_vicinity(data_dict, state_var, behaviour):
    all_vicinities = []
    for simulation in data_dict.values():
        df_var = simulation[state_var]
        size = df_var.shape[1]
        matrix = np.empty((0, size))
        for windowA in range(size):
            if behaviour == "seekers":
                mask = simulation['SimpleB'].iloc[:, windowA] != 1
            elif behaviour == "treated":
                mask = simulation['T'].iloc[:, windowA] == 0
            else:
                mask = pd.Series([False] * df_var.shape[0])

            vicinity = df_var.mask(mask, other=np.nan).mean(axis=0)
            matrix = np.concatenate((matrix, np.array([vicinity])), axis=0)
        all_vicinities.append(matrix)
    return np.array(all_vicinities)

# --- PLOTTING HELPERS ---

def plot_temporal_series(ax, data, label, color, linestyle, y_label, title=None, maxy=None):
    mean = np.nanmean(data, axis=0)
    quantiles = np.nanquantile(data, q=[0.05, 0.95], axis=0)
    x = range(len(mean))

    ax.plot(x, mean, label=label, linestyle=linestyle, color=color)
    ax.errorbar(x, mean, yerr=np.abs(quantiles - mean), fmt=',', ecolor=color, elinewidth=0.2)
    
    ax.set_xlabel("Time (Cycles)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    if title: ax.set_title(title, fontsize=12)
    if maxy: ax.set_ylim(top=maxy)
    ax.legend(fontsize=8)

def plot_histogram(ax, error_data, title, color, var='H', xlab="Unserved Needs", ylab="Patients", hist_label=None):
    final_h = np.array([d[var].iloc[:, -1] for d in error_data.values()])
    counts_list = [np.histogram(row, bins=20)[0] for row in final_h]
    mean_counts = np.mean(counts_list, axis=0)
    quantiles = np.quantile(counts_list, q=[0.05, 0.95], axis=0)
    edges = np.histogram(final_h[0], bins=20)[1]

    midbin = (edges[1] - edges[0]) / 2
    ax.stairs(mean_counts, edges, fill=True, color=color, label=hist_label)
    ax.errorbar(edges[:-1] + midbin, mean_counts, yerr=np.abs(quantiles - mean_counts), fmt='.', color='black')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run Detailed Model Exploration Plots")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--output-root", type=str, required=True, help="Root folder for experiment outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    
    global_settings = config["global_settings"]
    experiment_settings = config["experiment_settings"]
    baseline_parameters = config["baseline_parameters"]

    # Create run output directory
    run_dir = setup_output_dir(args.output_root)
    print(f"All outputs will be saved in: {run_dir}")

    # Save configuration snapshot
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 1. Load Baseline Parameters
    schedules = experiment_settings["treatments"]
    chosen_lambda = experiment_settings["lambda_to_run"]

    # Generate grid
    grid_rows = []
    simulation_metadata = []

    initial_seed = experiment_settings.get("initial_seed", 62920828945)
    reps = global_settings.get("reps", 10)
    reproduce_line = global_settings.get("reproduce_line", True)
    state_vars = global_settings["state_variables"]

    print("Generating simulation grid...")
    for schedule in schedules:
        params = baseline_parameters.copy()
        params['Pi'] = schedule
        params['reproduce_line'] = reproduce_line
        
        for rep in range(reps):
            seed = int(initial_seed + rep * 1485)
            sim_params = params.copy()
            sim_params['seed'] = seed
            sim_params['OBS_PERIOD'] = experiment_settings['OBS_PERIOD']
            sim_params['varsigma'] = experiment_settings['varsigma']
            sim_params['fixed_kappa'] = experiment_settings['fixed_kappa']
            sim_params['prioritization_granularity'] = experiment_settings.get('prioritization_granularity', -1)
            sim_params['obsPerformance'] = True
            
            # Enable observation flags for required state variables
            for var in state_vars:
                sim_params[f"obs{var}"] = True
                
            sim_params['PROVIDER_INIT'] = 'applyFixed'
            sim_params['PATIENT_INIT'] = 'applyFixed'
            
            sim_params.pop('seeds', None)
            sim_params.pop('H', None)
            sim_params.pop('Fitness', None)
            
            grid_rows.append(sim_params)
            simulation_metadata.append({
                "schedule": schedule,
                "rep": rep
            })

    grid_df = pd.DataFrame(grid_rows)
    print(f"Total simulations to run: {len(grid_df)}")

    # 2. Start Java Server and Execute Batch
    engine_path = global_settings["engine_path"]
    port = global_settings.get("port", 8383)
    
    print(f"Starting Java ABM server once using {engine_path} on port {port}...")
    c = Client(PORT=port, ENGINE_PATH=engine_path)
    c.start_server()

    try:
        print("Sending simulations batch to server...")
        results = c.socket_with_model_paramGrid_2(
            gridParameters=grid_df,
            PORT=port,
            ComputeErrors=1,
            batch_size=global_settings.get("batch_size", 100)
        )
    except Exception as e:
        print(f"Error executing simulations batch: {e}")
        return

    # Process and restructure results to match legacy format
    # Legacy format: mech_data[schedule][rep] = dict containing variables as DataFrames
    mech_data = {s: {} for s in schedules}
    
    for idx, meta in enumerate(simulation_metadata):
        res = results.get(idx)
        if res is None:
            continue
            
        schedule = meta["schedule"]
        rep = meta["rep"]
        
        rep_dict = {}
        for var in state_vars:
            if var in res:
                # Format as a dataframe
                rep_dict[var] = pd.DataFrame(res[var])
            else:
                # Fallback to an empty dataframe or handle list structure
                rep_dict[var] = pd.DataFrame()
        
        mech_data[schedule][rep] = rep_dict

    # 3. Setup Figures
    fig_delivery, ax_delivery = plt.subplots(figsize=(12, 7))
    fig_health, ax_health = plt.subplots(figsize=(12, 7))
    fig_seeking, ax_seeking = plt.subplots(figsize=(12, 7))
    fig_exp_corr, ax_exp_corr = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_diagon, ax_diagon = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_focus, ax_focus = plt.subplots(figsize=(18, 7))
    
    fig_mosaic = plt.figure(layout="constrained", figsize=(12, 7))
    axd = fig_mosaic.subplot_mosaic(
        """
        LB
        LR
        LN
        """,
        gridspec_kw=dict(width_ratios=[1.3, 1])
    )
    
    policy_colors = experiment_settings.get("policy_colors", {})
    policy_labels = experiment_settings.get("policy_labels", {})
    bound = experiment_settings["bound"]

    # 4. Generate Plot Series
    print("Generating plots...")
    for schedule in schedules:
        if schedule not in mech_data or not mech_data[schedule]:
            continue
        
        color = policy_colors.get(schedule, "black")
        label = policy_labels.get(schedule, schedule.upper())
        data_dict = mech_data[schedule]

        # 1. Delivery of Treatments
        try:
            treat_data = np.array([d['Performance'].values for d in data_dict.values() if not d['Performance'].empty]).mean(axis=1)
            treat_data[treat_data == 0] = np.nan
        except Exception:
            treat_data = np.array([np.nanmean(np.where(d['T'].values > 0, d['T'].values, np.nan), axis=0) for d in data_dict.values()])

        plot_temporal_series(ax=ax_delivery, data=treat_data, label=label, color=color, linestyle='solid',
                             title='Delivery of Treatments', y_label='Average Needs Solved per Appointment')
        ax_delivery.grid(True, linestyle=':', alpha=0.6)

        # 2. Progression of Diseases
        health_data = np.array([d['H'].values for d in data_dict.values()]).mean(axis=1)
        plot_temporal_series(ax=ax_health, data=health_data, label=label, color=color, linestyle='solid',
                             title='Progression of Diseases', y_label='Average Health Problems per Patient')
        
        plot_temporal_series(ax=axd['L'], data=health_data, label=label, color=color, linestyle='solid',
                             title='Progression of Diseases', y_label='Average Health Problems per Patient')

        # 3. Care Seeking Behaviour
        seeking_data = care_seeking(data_dict)
        plot_temporal_series(ax=ax_seeking, data=seeking_data, label=label, color=color, linestyle='solid',
                             title='Care Seeking Behaviour', y_label='Proportion of the Population Seeking Care')

        # 4. Expectations
        exp_lower = get_average_Exp_bound(data_dict, bound, lower=True)
        plot_temporal_series(ax=ax_exp_corr[1], data=exp_lower, label=f'{label}, mean expectations (lower need)', color=color,
                             linestyle='solid', title='Expectations in Lower Needs Patients',
                             y_label=f'Mean Expectations in the lower {bound:.2f} Needs Quantile')
        
        corrMaxExp = corrExp_Health(data_dict, expectationType='MaxExp')
        plot_temporal_series(ax=ax_exp_corr[0], data=np.array(corrMaxExp), label=f'{label}, correlation between expectations and needs', color=color, linestyle='solid',
                             title='Correlation between Expectations and Needs at the Population Level', y_label='Pearson correlation of Needs and Expectations')

        # 5. Focus Needs
        AllNeed = np.array([d['N'].values for d in data_dict.values()])
        AllTreat = np.array([d['T'].values for d in data_dict.values()])
        TreatmentNeed = np.where(AllTreat > 0, AllNeed, np.nan)
        q5, q25, q50, q75, q95 = np.nanquantile(TreatmentNeed, q=[0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
        xs = np.arange(q25.shape[1])
        alpha = 0.4 if schedule == 'need' else 0.2
        ax_focus.fill_between(xs, q25.mean(0), q75.mean(0), color=color, alpha=alpha, label=f'{label}, Needs Median and IQR')
        ax_focus.plot(xs, q25.mean(0), color=color, linestyle='--', linewidth=1, alpha=0.8)
        ax_focus.plot(xs, q75.mean(0), color=color, linestyle='--', linewidth=1, alpha=0.8)
        ax_focus.plot(xs, q50.mean(0), color=color, linestyle='solid', linewidth=1.5, alpha=1)
        ax_focus.plot(xs, q5.mean(0), color=color, linestyle='dotted', linewidth=1, alpha=0.8)
        ax_focus.plot(xs, q95.mean(0), color=color, linestyle='dotted', linewidth=1, alpha=0.8)

        # 6. Jaccard next cycle similarity
        seekJac = np.array(compute_jaccard(data_dict, 'SimpleB'))
        treatJac = np.array(compute_jaccard(data_dict, 'T'))
        
        # diagonal offset 1 average across reps
        seekJacDiag = np.array([m.diagonal(offset=1) for m in seekJac])
        treatJacDiag = np.array([m.diagonal(offset=1) for m in treatJac])
        
        plot_temporal_series(ax=ax_diagon[1], data=seekJacDiag, color=color, label=f'{label}, next cycle similarity', linestyle='solid', title='', y_label='')
        plot_temporal_series(ax=ax_diagon[0], data=treatJacDiag, color=color, label=f'{label}, next cycle similarity', linestyle='solid', title='', y_label='')

        # 7. Histograms on right of mosaic (end of simulation)
        ax_idx = 'B' if schedule == 'basal' else ('R' if schedule == 'risk' else 'N')
        plot_histogram(ax=axd[ax_idx], error_data=data_dict, title=f'{label} (End)', color=color, xlab='Disease Progression', hist_label=label)

    # 5. Finalize Figures and Save
    print(f"Saving exploration plots to: {run_dir}")
    
    fig_delivery.savefig(os.path.join(run_dir, 'delivery.png'), dpi=300)
    fig_health.savefig(os.path.join(run_dir, 'health.png'), dpi=300)
    fig_seeking.savefig(os.path.join(run_dir, 'seeking.png'), dpi=300)
    fig_exp_corr.savefig(os.path.join(run_dir, 'expectations_correlations.png'), dpi=300)
    fig_diagon.savefig(os.path.join(run_dir, 'diagonals.png'), dpi=300)
    fig_focus.savefig(os.path.join(run_dir, 'focus.png'), dpi=300)
    fig_mosaic.savefig(os.path.join(run_dir, 'mosaic.png'), dpi=300)
    
    plt.close('all')
    print("All plots saved successfully.")
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()
