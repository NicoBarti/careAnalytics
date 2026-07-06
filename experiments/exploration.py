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
from products.turnover.alluvial import (plot_alluvial_deciles, compute_churning, 
                                        compute_mutual_information, 
                                        compute_discretized_mutual_information,
                                        compute_normalized_discretized_mutual_information,
                                        compute_kl_divergence,
                                        compute_flow_entropy,
                                        compute_severity_entropy)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_output_dir(output_root, param_name=None, param_value=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if param_name is not None and param_value is not None:
        run_dir = os.path.join(output_root, f"exploration_run_{param_name}_{param_value}_{timestamp}")
    else:
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

def compute_vicinity(data_dict, state_var, behaviour, all_replicates=False):
    """Computes average state variable based on patient behaviour mask.
    
    Args:
        data_dict (dict): Dictionary of simulation runs.
        state_var (str): The state variable to analyze.
        behaviour (str): "seekers" or "treated" to define the mask.
        all_replicates (bool): If True, averages over all replicates. If False, uses only the first.
    """
    def _get_vicinity(simulation):
        size = simulation[state_var].shape[1]
        matrix = np.empty((0, size))
        for windowA in range(size):
            if behaviour == "seekers":
                mask = simulation['SimpleB'].iloc[:, windowA] != 1
            elif behaviour == "treated":
                mask = simulation['T'].iloc[:, windowA] == 0
            else:
                mask = pd.Series([False] * simulation[state_var].shape[0])

            vicinity = simulation[state_var].mask(mask, other=np.nan).mean(axis=0)
            matrix = np.concatenate((matrix, np.array([vicinity])), axis=0)
        return matrix

    if not all_replicates:
        return _get_vicinity(data_dict[next(iter(data_dict))])

    all_vicinities = [_get_vicinity(sim) for sim in data_dict.values()]
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

def plot_histogram(ax, error_data, title, color, var='H', xlab="Unserved Needs", ylab="Patients", hist_label=None, bins=20, draw_deciles=False):
    final_h = np.array([d[var].iloc[:, -1] for d in error_data.values()])
    counts_list = [np.histogram(row, bins=bins)[0] for row in final_h]
    mean_counts = np.mean(counts_list, axis=0)
    quantiles = np.quantile(counts_list, q=[0.05, 0.95], axis=0)
    edges = np.histogram(final_h[0], bins=bins)[1]

    midbin = (edges[1] - edges[0]) / 2
    ax.stairs(mean_counts, edges, fill=True, color=color, label=hist_label)
    ax.errorbar(edges[:-1] + midbin, mean_counts, yerr=np.abs(quantiles - mean_counts), fmt='.', color='black')

    if draw_deciles:
        deciles = np.percentile(final_h.flatten(), np.arange(10, 100, 10))
        for idx, dec in enumerate(deciles):
            ax.axvline(x=dec, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Decile' if idx == 0 else "")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run Detailed Model Exploration Plots")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--output-root", type=str, default=None, help="Root folder for experiment outputs")
    parser.add_argument("--param-index", type=int, default=None, help="Index of parameter sweep configuration to use from sensitivity grid")
    args = parser.parse_args()

    config = load_config(args.config)
    
    global_settings = config["global_settings"]
    experiment_settings = config["experiment_settings"]
    baseline_parameters = config["baseline_parameters"]

    # 1. Parse and apply parameter override from grid if param-index is requested
    sensitivity_configs = experiment_settings.get("sensitivity_configs")
    selected_param = None
    selected_val = None
    
    if sensitivity_configs and args.param_index is not None:
        grid_points = []
        for param in sorted(sensitivity_configs.keys()):
            for val in sorted(sensitivity_configs[param]):
                grid_points.append((param, val))
                
        if args.param_index < 0 or args.param_index >= len(grid_points):
            print(f"Error: --param-index {args.param_index} is out of bounds. "
                  f"Available indices are 0 to {len(grid_points) - 1}.")
            return
            
        selected_param, selected_val = grid_points[args.param_index]
        print(f"Selected parameter override from grid (index {args.param_index}): "
              f"{selected_param} = {selected_val}")
        
        # Update baseline parameters with the override and remove the sensitivity sweep grid
        config["baseline_parameters"][selected_param] = selected_val
        if "sensitivity_configs" in config["experiment_settings"]:
            config["experiment_settings"].pop("sensitivity_configs", None)

    # Resolve output directory 
    output_root = args.output_root
    if output_root is None:
        output_root = os.path.dirname(os.path.abspath(args.config))

    # Create run output directory
    run_dir = setup_output_dir(output_root, param_name=selected_param, param_value=selected_val)
    print(f"All outputs will be saved in: {run_dir}")

    # Save configuration snapshot
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 2. Load Baseline Parameters
    schedules = experiment_settings.get("treatments", list(experiment_settings.get("policy_colors", {}).keys()))
    chosen_lambda = experiment_settings.get("lambda_to_run", 4.0)

    # Generate grid
    grid_rows = []
    simulation_metadata = []

    initial_seed = experiment_settings.get("initial_seed", 62920828945)
    reps = global_settings.get("reps", 10)
    reproduce_line = global_settings.get("reproduce_line", True)
    state_vars = list(global_settings["state_variables"])
    REQUIRED_VARS = ["H", "N", "T", "SimpleB", "stepPerformance", "MaxExp", "SimpleE", "Delta", "SimpleC", "Disease", "ExpNoise"]
    for var in REQUIRED_VARS:
        if var not in state_vars:
            state_vars.append(var)

    print("Generating simulation grid...")
    for schedule in schedules:
        params = baseline_parameters.copy()
        params['Pi'] = schedule
        params['reproduce_line'] = reproduce_line
        
        for rep in range(reps):
            seed = int(initial_seed + rep * 1485)
            sim_params = params.copy()
            sim_params['seed'] = seed
            sim_params['OBS_PERIOD'] = experiment_settings.get('OBS_PERIOD', 30)
            sim_params['varsigma'] = experiment_settings.get('varsigma', 300)
            sim_params['fixed_kappa'] = experiment_settings.get('fixed_kappa', params.get('fixed_kappa', 0.5))
            sim_params['prioritization_granularity'] = experiment_settings.get('prioritization_granularity', params.get('prioritization_granularity', -1))
            
            # Apply parameter index override if active
            if selected_param is not None:
                sim_params[selected_param] = selected_val
            
            # Enable observation flags for required state variables
            for var in state_vars:
                if var == "stepPerformance":
                    sim_params['stepPerformance'] = True
                else:
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
        # Include fallback if connection with server fails and engine_path contains users/Nico
        if "/users/Nico" in engine_path or "/Users/Nico" in engine_path:
            fallback_engine_path = engine_path.replace("/users/Nico", "/Users/nicolasbarticevic").replace("/Users/Nico", "/Users/nicolasbarticevic")
            print(f"Retrying with fallback engine path: {fallback_engine_path}...")
            c = Client(PORT=port, ENGINE_PATH=fallback_engine_path)
            c.start_server()
            try:
                print("Sending simulations batch to server (retry)...")
                results = c.socket_with_model_paramGrid_2(
                    gridParameters=grid_df,
                    PORT=port,
                    ComputeErrors=1,
                    batch_size=global_settings.get("batch_size", 100)
                )
            except Exception as e_retry:
                print(f"Error executing simulations batch on retry: {e_retry}")
                return
        else:
            return

    # Process and restructure results to match legacy format
    # Legacy format: mech_data[schedule][rep] = dict containing variables as DataFrames
    mech_data = {s: {} for s in schedules}
    windows = []
    
    for idx, meta in enumerate(simulation_metadata):
        res = results.get(idx)
        if res is None:
            continue
            
        schedule = meta["schedule"]
        rep = meta["rep"]
        
        rep_dict = {}
        windows = res.get("windows", [])
        columns = [str(w) for w in windows] if windows else None
        for var in state_vars:
            if var in res:
                # Format as a dataframe and assign timestep names as columns if 2D
                val_data = res[var]
                if isinstance(val_data, list) and len(val_data) > 0 and isinstance(val_data[0], list):
                    if columns and len(columns) == len(val_data[0]):
                        rep_dict[var] = pd.DataFrame(val_data, columns=columns)
                    else:
                        rep_dict[var] = pd.DataFrame(val_data)
                else:
                    rep_dict[var] = pd.DataFrame(val_data)
            else:
                # Fallback to an empty dataframe or handle list structure
                rep_dict[var] = pd.DataFrame()
        
        mech_data[schedule][rep] = rep_dict

    # Resolve windows and target cycles for row grid
    if not windows:
        for s in schedules:
            if s in mech_data and mech_data[s]:
                first_rep = next(iter(mech_data[s].values()))
                if 'H' in first_rep and not first_rep['H'].empty:
                    windows = [int(c) for c in first_rep['H'].columns]
                    break
    if not windows:
        windows = list(range(0, 301, 30))

    target_cycles = [5, 60, 170]
    row_indices = []
    for target in target_cycles:
        closest_idx = int(np.argmin([abs(int(w) - target) for w in windows]))
        row_indices.append(closest_idx)
    row_indices = sorted(list(set(row_indices)))
    while len(row_indices) < 3:
        for i in range(len(windows)):
            if i not in row_indices:
                row_indices.append(i)
                row_indices = sorted(row_indices)
                break

    # 3. Setup Figures for Standard Plots
    print("Setting up figures for standard plots...")
    fig_delivery, ax_delivery = plt.subplots(figsize=(12, 7))
    fig_health, ax_health = plt.subplots(figsize=(12, 7))
    fig_seeking, ax_seeking = plt.subplots(figsize=(12, 7))
    fig_exp_corr, ax_exp_corr = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_diagon, ax_diagon = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    fig_focus, ax_focus = plt.subplots(figsize=(18, 7))
    
    fig_mosaic = plt.figure(layout="constrained", figsize=(12, 7))
    axd_mosaic = fig_mosaic.subplot_mosaic(
        """
        LB
        LR
        LN
        """,
        gridspec_kw=dict(width_ratios=[1.3, 1])
    )
    
    policy_colors = experiment_settings.get("policy_colors", {})
    policy_labels = experiment_settings.get("policy_labels", {})
    bound = experiment_settings.get("bound", 0.2857)

    # 4. Generate Plot Series
    print("Generating standard plots...")
    for schedule in schedules:
        if schedule not in mech_data or not mech_data[schedule]:
            continue
        
        color = policy_colors.get(schedule, "black")
        label = policy_labels.get(schedule, schedule.upper())
        data_dict = mech_data[schedule]

        # 1. Delivery of Treatments
        try:
            treat_data = np.array([d['stepPerformance'].values.flatten() for d in data_dict.values() if not d['stepPerformance'].empty])
            treat_data[treat_data == 0] = np.nan
        except Exception:
            treat_data = np.array([np.nanmean(np.where(d['T'].values > 0, d['T'].values, np.nan), axis=0) for d in data_dict.values()])

        plot_temporal_series(ax=ax_delivery, data=treat_data, label=label, color=color, linestyle='solid',
                             title='Delivery of Treatments', y_label='Performance per Cycle')
        ax_delivery.grid(True, linestyle=':', alpha=0.6)

        # 2. Progression of Diseases
        health_data = np.array([d['H'].values for d in data_dict.values()]).mean(axis=1)
        plot_temporal_series(ax=ax_health, data=health_data, label=label, color=color, linestyle='solid',
                             title='Progression of Diseases', y_label='Average Health Problems per Patient')
        
        plot_temporal_series(ax=axd_mosaic['L'], data=health_data, label=label, color=color, linestyle='solid',
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
        
        plot_temporal_series(ax=ax_diagon[0], data=seekJacDiag, color=color, label=f'{label}, next cycle similarity', linestyle='solid', title='Care Seeking Similarity', y_label=r'\% of same patients seeking care in previous cycle')
        plot_temporal_series(ax=ax_diagon[1], data=treatJacDiag, color=color, label=f'{label}, next cycle similarity', linestyle='solid', title='Treatment Similarity', y_label=r'\% of same patients receiving treatment in previous cycle')

        # 7. Histograms on right of mosaic (end of simulation)
        ax_idx = 'B' if schedule == 'basal' else ('R' if schedule == 'risk' else 'N')
        plot_histogram(ax=axd_mosaic[ax_idx], error_data=data_dict, title=f'{label} (End)', color=color, xlab='Disease Progression', hist_label=label)

    # Save original figures
    print(f"Saving standard exploration plots to: {run_dir}")
    fig_delivery.savefig(os.path.join(run_dir, 'delivery.png'), dpi=300)
    fig_health.savefig(os.path.join(run_dir, 'health.png'), dpi=300)
    fig_seeking.savefig(os.path.join(run_dir, 'seeking.png'), dpi=300)
    fig_exp_corr.savefig(os.path.join(run_dir, 'expectations_correlations.png'), dpi=300)
    fig_diagon.savefig(os.path.join(run_dir, 'diagonals.png'), dpi=300)
    fig_focus.savefig(os.path.join(run_dir, 'focus.png'), dpi=300)
    fig_mosaic.savefig(os.path.join(run_dir, 'mosaic.png'), dpi=300)
    
    # 5. Generate Alluvial Diagrams and Flow Metrics
    print("Generating alluvial and flow metric plots...")
    
    num_deciles = experiment_settings.get("num_deciles", 
                                           experiment_settings.get("num_quantiles", 
                                                                    experiment_settings.get("bins", 10)))
    entropy_bins = experiment_settings.get("entropy_bins", 
                                            experiment_settings.get("num_quantiles", 
                                                                     experiment_settings.get("bins", 10)))
    num_observations = experiment_settings.get("num_observations", 11)
    
    # Select evenly spaced timesteps from available windows
    obs_indices = np.linspace(0, len(windows) - 1, min(num_observations, len(windows)))
    timesteps = [int(windows[int(round(idx))]) for idx in obs_indices]
    
    # 5.1 Alluvial Diagrams per policy
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        rep_data = list(data_dict.values())[0]
        H_df = rep_data.get('H')
        Delta_df = rep_data.get('Delta')
        
        if H_df is not None and not H_df.empty:
            output_path = os.path.join(run_dir, f"alluvial_{schedule}.png")
            label = policy_labels.get(schedule, schedule.upper())
            title = f"Health Deciles Alluvial Diagram (Colored by End Quantile Delta) - {label}"
            
            color_by_delta = Delta_df is not None and not Delta_df.empty
            
            plot_alluvial_deciles(
                H_df=H_df,
                Delta_df=Delta_df if color_by_delta else pd.DataFrame(0, index=H_df.index, columns=['0']),
                title=title,
                save_path=output_path,
                num_deciles=num_deciles,
                num_observations=len(timesteps),
                varsigma=experiment_settings.get('varsigma', 300),
                start_time=timesteps[0],
                end_time=timesteps[-1],
                color_by_delta=color_by_delta
            )

    # 5.2 Patient Churning Rates over time
    churning_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_rates = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            if H_df is not None and not H_df.empty:
                rates = compute_churning(H_df, entropy_bins, timesteps)
                all_reps_rates.append(rates)
        if all_reps_rates:
            churning_results[schedule] = np.mean(all_reps_rates, axis=0)
            
    if churning_results:
        transition_times = timesteps[1:]
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in churning_results:
                continue
            rates = churning_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = transition_times[:len(rates)]
            plt.plot(x_vals, rates, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        plt.title("Patient Health Deciles Churning Rate Over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle (Transition End Time)", fontsize=12, labelpad=10)
        plt.ylabel("Churning Rate (Proportion Changing Quantile)", fontsize=12, labelpad=10)
        plt.ylim(-0.05, 1.05)
        plt.xticks(transition_times)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=11)
        plt.tight_layout()
        
        churn_plot_path = os.path.join(run_dir, "churning_rate.png")
        plt.savefig(churn_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Churning rate plot saved to {churn_plot_path}")

    # 5.2.1 Shannon Entropy of Transition Flows over time
    flow_entropy_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_entropies = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            if H_df is not None and not H_df.empty:
                entropies = compute_flow_entropy(H_df, entropy_bins, timesteps)
                if entropies is not None:
                    all_reps_entropies.append(entropies)
        if all_reps_entropies:
            flow_entropy_results[schedule] = np.mean(all_reps_entropies, axis=0)
            
    if flow_entropy_results:
        transition_times = timesteps[1:]
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in flow_entropy_results:
                continue
            entropies = flow_entropy_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = transition_times[:len(entropies)]
            plt.plot(x_vals, entropies, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        # Reference lines for theoretical min/max entropy bounds
        min_entropy = np.log2(entropy_bins)
        max_entropy = 2 * np.log2(entropy_bins)
        plt.axhline(y=min_entropy, color='#cbd5e0', linestyle='--', linewidth=1.5, label='Min Theoretical Entropy (No Churn)')
        plt.axhline(y=max_entropy, color='#feb2b2', linestyle='--', linewidth=1.5, label='Max Theoretical Entropy (Full Shuffling)')
        
        plt.title("Transition Flow Shannon Entropy Over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle (Transition End Time)", fontsize=12, labelpad=10)
        plt.ylabel("Shannon Entropy (Bits)", fontsize=12, labelpad=10)
        plt.ylim(min_entropy - 0.2, max_entropy + 0.2)
        plt.xticks(transition_times)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=11)
        plt.tight_layout()
        
        flow_entropy_plot_path = os.path.join(run_dir, "flow_entropy.png")
        plt.savefig(flow_entropy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Flow entropy plot saved to {flow_entropy_plot_path}")

    # 5.3 KSG Mutual Information between H and Severity (Disease) over time
    mi_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_mis = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            Disease_df = rep_data.get('Disease')
            if H_df is not None and not H_df.empty and Disease_df is not None and not Disease_df.empty:
                mi_vals = compute_mutual_information(H_df, Disease_df, timesteps)
                if mi_vals is not None:
                    all_reps_mis.append(mi_vals)
        if all_reps_mis:
            mi_results[schedule] = np.mean(all_reps_mis, axis=0)
            
    if mi_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in mi_results:
                continue
            mi_vals = mi_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = timesteps[:len(mi_vals)]
            plt.plot(x_vals, mi_vals, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        plt.title(r"KSG Mutual Information between Health Status (H) and Severity ($\delta$) over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("KSG Mutual Information (Nats)", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        mi_plot_path = os.path.join(run_dir, "ksg_mutual_information_H_severity.png")
        plt.savefig(mi_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"KSG mutual information plot saved to {mi_plot_path}")

    # 5.4 Discretized Mutual Information between H and Severity (Disease) over time
    disc_mi_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_disc_mis = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            Disease_df = rep_data.get('Disease')
            if H_df is not None and not H_df.empty and Disease_df is not None and not Disease_df.empty:
                disc_mi_vals = compute_discretized_mutual_information(H_df, Disease_df, timesteps, bins=entropy_bins)
                if disc_mi_vals is not None:
                    all_reps_disc_mis.append(disc_mi_vals)
        if all_reps_disc_mis:
            disc_mi_results[schedule] = np.mean(all_reps_disc_mis, axis=0)
            
    if disc_mi_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in disc_mi_results:
                continue
            disc_mi_vals = disc_mi_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = timesteps[:len(disc_mi_vals)]
            plt.plot(x_vals, disc_mi_vals, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        plt.title(f"Discretized Mutual Information between Health Status (H) and Severity ($\\delta$) over Time ({entropy_bins} Bins)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("Discretized Mutual Information (Nats)", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        disc_mi_plot_path = os.path.join(run_dir, "discretized_mutual_information_H_severity.png")
        plt.savefig(disc_mi_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Discretized mutual information plot saved to {disc_mi_plot_path}")

    # 5.5 Normalized Discretized Mutual Information between H and Severity (Disease) over time
    norm_mi_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_norm_mis = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            Disease_df = rep_data.get('Disease')
            if H_df is not None and not H_df.empty and Disease_df is not None and not Disease_df.empty:
                norm_mi_vals = compute_normalized_discretized_mutual_information(H_df, Disease_df, timesteps, bins=entropy_bins)
                if norm_mi_vals is not None:
                    all_reps_norm_mis.append(norm_mi_vals)
        if all_reps_norm_mis:
            norm_mi_results[schedule] = np.mean(all_reps_norm_mis, axis=0)
            
    if norm_mi_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in norm_mi_results:
                continue
            norm_mi_vals = norm_mi_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = timesteps[:len(norm_mi_vals)]
            plt.plot(x_vals, norm_mi_vals, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        plt.title(f"Normalized Discretized Mutual Information between Health Status (H) and Severity ($\\delta$) over Time ({entropy_bins} Bins)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("Normalized Discretized Mutual Information", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        norm_mi_plot_path = os.path.join(run_dir, "normalized_discretized_mutual_information_H_severity.png")
        plt.savefig(norm_mi_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normalized discretized mutual information plot saved to {norm_mi_plot_path}")

    # 5.6 Relative Entropy (KL Divergence) of Severity (Delta) across Deciles over time
    kl_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_kls = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            Delta_df = rep_data.get('Delta')
            if H_df is not None and not H_df.empty and Delta_df is not None and not Delta_df.empty:
                kl_vals = compute_kl_divergence(H_df, Delta_df, entropy_bins, timesteps)
                if kl_vals is not None:
                    all_reps_kls.append(kl_vals)
        if all_reps_kls:
            kl_results[schedule] = np.mean(all_reps_kls, axis=0)
            
    if kl_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in kl_results:
                continue
            kl_vals = kl_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = timesteps[:len(kl_vals)]
            plt.plot(x_vals, kl_vals, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        label_prefix = "Decile" if entropy_bins == 10 else f"{entropy_bins}-Quantile"
        plt.title(f"Relative Entropy (KL Divergence) of Severity across {label_prefix}s over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("KL Divergence from Time 0 (Nats)", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        kl_plot_path = os.path.join(run_dir, "kl_divergence_severity.png")
        plt.savefig(kl_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"KL divergence plot saved to {kl_plot_path}")

    # 5.7 Shannon Entropy (self-information) of Severity (Delta) across Deciles over time
    entropy_severity_results = {}
    for schedule, data_dict in mech_data.items():
        if not data_dict:
            continue
        all_reps_entropies = []
        for rep_data in data_dict.values():
            H_df = rep_data.get('H')
            Delta_df = rep_data.get('Delta')
            if H_df is not None and not H_df.empty and Delta_df is not None and not Delta_df.empty:
                entropy_vals = compute_severity_entropy(H_df, Delta_df, entropy_bins, timesteps)
                if entropy_vals is not None:
                    all_reps_entropies.append(entropy_vals)
        if all_reps_entropies:
            entropy_severity_results[schedule] = np.mean(all_reps_entropies, axis=0)
            
    if entropy_severity_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        for schedule in ['basal', 'risk', 'need']:
            if schedule not in entropy_severity_results:
                continue
            entropy_vals = entropy_severity_results[schedule]
            color = policy_colors.get(schedule, "black")
            label = policy_labels.get(schedule, schedule.upper())
            
            style_marker = 'o' if schedule == 'need' else ('s' if schedule == 'risk' else '^')
            style_line = '-' if schedule == 'need' else ('--' if schedule == 'risk' else ':')
            
            x_vals = timesteps[:len(entropy_vals)]
            plt.plot(x_vals, entropy_vals, label=label, color=color, 
                     marker=style_marker, linestyle=style_line, linewidth=2,
                     markersize=8, alpha=0.9)
            
        label_prefix = "Decile" if entropy_bins == 10 else f"{entropy_bins}-Quantile"
        
        # Max theoretical discrete severity entropy is log(20) because of 20 severity bins
        max_entropy = np.log(20)
        plt.axhline(y=max_entropy, color='#feb2b2', linestyle='--', linewidth=1.5, label='Max Theoretical Entropy (Uniform)')
        
        plt.title(f"Shannon Entropy (Self-Information) of Severity across {label_prefix}s over Time", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("Shannon Entropy (Nats)", fontsize=12, labelpad=10)
        plt.ylim(0.0, max_entropy + 0.1)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        entropy_plot_path = os.path.join(run_dir, "severity_entropy.png")
        plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Shannon Entropy (self-information) of Severity plot saved to {entropy_plot_path}")
    # 6. Setup Figures and Generate Complex Mosaic Plots
    print("Generating Complex Mosaic Plots...")
    from products.outcomesMatrix.complex_plotter import complexAxeDict, populate_axe

    for schedule in schedules:
        if schedule not in mech_data or not mech_data[schedule]:
            continue
        
        reps_list = list(mech_data[schedule].keys())
        if not reps_list:
            continue

        label = policy_labels.get(schedule, schedule.upper())
        print(f"Generating Complex Mosaic Plot for {label}...")

        # Combine results across all replications by concatenating patient rows
        combined_dd = {}
        for var in state_vars:
            # Map MaxExp to E as expected by complex_plotter
            target_var = 'E' if var == 'MaxExp' else var
            rep_dfs = []
            for rep in reps_list:
                if var in mech_data[schedule][rep]:
                    df = mech_data[schedule][rep][var]
                    if not df.empty:
                        rep_dfs.append(df)
            if rep_dfs:
                combined_dd[target_var] = pd.concat(rep_dfs, axis=0, ignore_index=True)
            else:
                combined_dd[target_var] = pd.DataFrame()

        # Generate the Complex Mosaic Plot
        last_col = combined_dd['H'].columns[-1]
        allData = np.quantile(np.array(combined_dd['H'][last_col]), q=[1])[0]

        axd = complexAxeDict()
        fig = plt.gcf()

        # Create sub_title displaying policy label and simulation parameters
        sub_title = f"Policy: {label} | Replications: {len(reps_list)} | Cycles: {last_col}"
        populate_axe(axd, dd=combined_dd, low_h_cut=0, high_h_cut=allData, sub_title=sub_title)

        fig.suptitle(f"Complex Mosaic Plot - {label}", fontsize=40, y=0.98)

        # Save the figure
        mosaic_path = os.path.join(run_dir, f"complex_mosaic_{schedule}.png")
        fig.savefig(mosaic_path, dpi=300)
        plt.close(fig)
        print(f"Complex Mosaic Plot for {label} saved to {mosaic_path}")

    # 7. Cleanup
    plt.close('all')
    print("All plots saved successfully.")
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()
