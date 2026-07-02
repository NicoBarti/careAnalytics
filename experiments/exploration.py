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
from products.turnover.alluvial import plot_alluvial_deciles, compute_churning, compute_flow_entropy

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

    # 3. Setup Figures and Generate Complex Mosaic Plots
    print("Generating Complex Mosaic Plots...")
    from products.outcomesMatrix.complex_plotter import complexAxeDict, populate_axe

    policy_labels = experiment_settings.get("policy_labels", {})

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

    # 4. Cleanup
    plt.close('all')
    print("All plots saved successfully.")
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()
