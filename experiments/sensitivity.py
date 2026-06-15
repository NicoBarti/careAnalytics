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

# Add the project root to sys.path to allow importing MyClasses
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MyClasses.client import Client

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_output_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"sensitivity_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis in Batches")
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

    # Save configuration snapshot to output directory
    with open(os.path.join(run_dir, "config_snapshot.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 1. Load Baseline parameters
    policies = list(experiment_settings.get("policy_colors", {}).keys())

    # Generate the parameter grid
    grid_rows = []
    simulation_metadata = [] # Keep track of which row corresponds to which target param/val/policy/rep

    reps = global_settings.get("reps", 1)
    reproduce_line = global_settings.get("reproduce_line", True)
    obs_period = experiment_settings["OBS_PERIOD"]
    initial_seed = experiment_settings.get("initial_seed", 62920828945)
    varsigma = experiment_settings.get("varsigma", 300)
    sensitivity_configs = experiment_settings["sensitivity_configs"]

    print("Generating simulation grid...")
    for target_param, test_values in sensitivity_configs.items():
        for policy in policies:
            baseline_params = baseline_parameters.copy()
            baseline_params['Pi'] = policy
            baseline_params['reproduce_line'] = reproduce_line
            
            for val in test_values:
                for rep in range(reps):
                    seed = int(initial_seed + rep * 1485)
                    sim_params = baseline_params.copy()
                    
                    # Update simulation params
                    sim_params['seed'] = seed
                    sim_params['OBS_PERIOD'] = obs_period
                    sim_params['varsigma'] = varsigma
                    sim_params[target_param] = val
                    
                    # Enforce observation flags required
                    sim_params['obsH'] = True
                    sim_params['PROVIDER_INIT'] = 'applyFixed'
                    sim_params['PATIENT_INIT'] = 'applyFixed'
                    
                    # Remove unwanted keys
                    sim_params.pop('seeds', None)
                    sim_params.pop('H', None)
                    sim_params.pop('Fitness', None)
                    
                    grid_rows.append(sim_params)
                    simulation_metadata.append({
                        "target_param": target_param,
                        "policy": policy,
                        "param_value": val,
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
    
    # 3. Process and Aggregate Results
    print("Processing results...")
    processed_rows = []
    
    for idx, meta in enumerate(simulation_metadata):
        res = results.get(idx)
        if res is None:
            print(f"Warning: No results found for simulation index {idx}")
            continue
        
        h_data = res.get("H", [])
        if not h_data:
            print(f"Warning: Missing 'H' state variable for index {idx}")
            continue
        
        h_arr = np.array(h_data, dtype=float)
        final_cycle_avg = np.mean(h_arr[:, -1])
        final_cycle_std = np.std(h_arr[:, -1])
        
        processed_rows.append({
            "target_param": meta["target_param"],
            "policy": meta["policy"],
            "param_value": meta["param_value"],
            "rep": meta["rep"],
            "mean_H": final_cycle_avg,
            "std_H": final_cycle_std
        })

    results_df = pd.DataFrame(processed_rows)
    # Save raw results
    results_df.to_csv(os.path.join(run_dir, "simulation_results.csv"), index=False)
    print("Simulation results saved to CSV.")

    # 4. Generate Plots
    print("Generating plots...")
    active_params = list(sensitivity_configs.keys())
    num_params = len(active_params)
    ncols = 3
    nrows = int(np.ceil(num_params / ncols))

    col_map = mpl.colormaps['plasma']
    policy_colors = experiment_settings.get("policy_colors", {})
    policy_labels = experiment_settings.get("policy_labels", {})
    param_labels = experiment_settings.get("param_labels", {})

    grid_fig_mean, grid_ax_mean = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    grid_fig_std, grid_ax_std = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    
    axes_mean_flat = grid_ax_mean.flatten() if nrows > 1 else np.array([grid_ax_mean]).flatten()
    axes_std_flat = grid_ax_std.flatten() if nrows > 1 else np.array([grid_ax_std]).flatten()

    for i in range(num_params, len(axes_mean_flat)):
        axes_mean_flat[i].set_visible(False)
        axes_std_flat[i].set_visible(False)

    for target_param in active_params:
        param_idx = active_params.index(target_param)
        ax_mean_g = axes_mean_flat[param_idx]
        ax_std_g = axes_std_flat[param_idx]
        
        fig_mean, ax_mean = plt.subplots(figsize=(12, 7), layout='constrained')
        fig_std, ax_std = plt.subplots(figsize=(12, 7), layout='constrained')

        param_df = results_df[results_df["target_param"] == target_param]
        test_values = sorted(param_df["param_value"].unique())
        param_label = param_labels.get(target_param, target_param)

        for policy in policies:
            policy_df = param_df[param_df["policy"] == policy]
            if policy_df.empty:
                continue

            # Group by param_val to aggregate over reps
            aggregated = policy_df.groupby("param_value").agg(
                mean_H_avg=("mean_H", "mean"),
                mean_H_q05=("mean_H", lambda x: x.quantile(0.05)),
                mean_H_q95=("mean_H", lambda x: x.quantile(0.95)),
                std_H_avg=("std_H", "mean"),
                std_H_q05=("std_H", lambda x: x.quantile(0.05)),
                std_H_q95=("std_H", lambda x: x.quantile(0.95))
            ).reindex(test_values)

            mean_vals = aggregated["mean_H_avg"].values
            mean_q05 = aggregated["mean_H_q05"].values
            mean_q95 = aggregated["mean_H_q95"].values

            std_vals = aggregated["std_H_avg"].values
            std_q05 = aggregated["std_H_q05"].values
            std_q95 = aggregated["std_H_q95"].values

            color = policy_colors.get(policy, "black")
            label = policy_labels.get(policy, policy.upper())

            # Plot mean H
            ax_mean.plot(test_values, mean_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_mean.errorbar(test_values, mean_vals, yerr=[mean_vals - mean_q05, mean_q95 - mean_vals],
                             fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)
            ax_mean_g.plot(test_values, mean_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_mean_g.errorbar(test_values, mean_vals, yerr=[mean_vals - mean_q05, mean_q95 - mean_vals],
                               fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)

            # Plot std H
            ax_std.plot(test_values, std_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_std.errorbar(test_values, std_vals, yerr=[std_vals - std_q05, std_q95 - std_vals],
                            fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)
            ax_std_g.plot(test_values, std_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_std_g.errorbar(test_values, std_vals, yerr=[std_vals - std_q05, std_q95 - std_vals],
                              fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)

        # Finalize individual and grid labels
        for ax, ylabel, title_prefix in [
            (ax_mean, 'Average Health Status', 'Mean'),
            (ax_mean_g, 'Average Health Status', 'Mean'),
            (ax_std, 'Standard Deviation of Health Status', 'Std Dev'),
            (ax_std_g, 'Standard Deviation of Health Status', 'Std Dev')
        ]:
            ax.set_xlabel(param_label, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Save individual plots
        fig_mean.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_mean.png"), dpi=300)
        fig_std.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_std.png"), dpi=300)
        plt.close(fig_mean)
        plt.close(fig_std)

    # Save grid plots
    grid_fig_mean.savefig(os.path.join(run_dir, "grid_sensitivity_mean.png"), dpi=300)
    grid_fig_std.savefig(os.path.join(run_dir, "grid_sensitivity_std.png"), dpi=300)
    plt.close(grid_fig_mean)
    plt.close(grid_fig_std)

    print("All plots generated and saved successfully.")

if __name__ == "__main__":
    main()
