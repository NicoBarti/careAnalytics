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
from products.turnover.alluvial import compute_sum_entropy_and_churning

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_output_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"sensitivity_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis in Batches")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--output-root", type=str, default=None, help="Root folder for experiment outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    
    global_settings = config["global_settings"]
    experiment_settings = config["experiment_settings"]
    baseline_parameters = config["baseline_parameters"]

    # Resolve output directory
    output_root = args.output_root
    if output_root is None:
        output_root = os.path.dirname(os.path.abspath(args.config))

    # Create run output directory
    run_dir = setup_output_dir(output_root)
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
    num_deciles = experiment_settings.get("num_deciles", 10)
    sensitivity_configs = experiment_settings["sensitivity_configs"]

    if obs_period >= varsigma:
        adjusted_obs = max(1, varsigma // 10)
        print(f"WARNING: OBS_PERIOD ({obs_period}) is >= varsigma ({varsigma}). "
              f"Flow metrics require intermediate observations. "
              f"Dynamically adjusting OBS_PERIOD to {adjusted_obs}.")
        obs_period = adjusted_obs

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
                    sim_params['stepPerformance'] = True
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
    
    # 3. Process and Aggregate Results
    print("Processing results...")
    processed_rows = []
    
    for idx, meta in enumerate(simulation_metadata):
        res = results.get(idx)
        if res is None:
            print(f"Warning: No results found for simulation index {idx} "
                  f"({meta['target_param']}={meta['param_value']}, policy={meta['policy']}, rep={meta['rep']})")
            continue
        
        h_data = res.get("H", [])
        if not h_data:
            print(f"Warning: Missing 'H' state variable for index {idx} "
                  f"({meta['target_param']}={meta['param_value']}, policy={meta['policy']}, rep={meta['rep']})")
            continue
        
        h_arr = np.array(h_data, dtype=float)
        final_cycle_avg = np.mean(h_arr[:, -1])
        final_cycle_std = np.std(h_arr[:, -1])
        
        # Compute quantile volatility and flow entropy
        windows = res.get("windows", [])
        if not windows:
            h_len = h_arr.shape[1]
            windows = [int(w * obs_period) for w in range(h_len)]
            
        H_df = pd.DataFrame(h_data, columns=[str(w) for w in windows])
        
        try:
            sum_flow_entropy, sum_quantile_volatility = compute_sum_entropy_and_churning(H_df, num_deciles, windows)
        except Exception as e:
            print(f"Warning: Failed to compute flow entropy/quantile volatility for index {idx}: {e}")
            sum_flow_entropy = 0.0
            sum_quantile_volatility = 0.0

        perf_data = res.get("stepPerformance", [])
        if not perf_data:
            print(f"Warning: Missing 'stepPerformance' state variable for index {idx} "
                  f"({meta['target_param']}={meta['param_value']}, policy={meta['policy']}, rep={meta['rep']})")
            total_performance = 0.0
        else:
            total_performance = float(np.sum(perf_data))

        processed_rows.append({
            "target_param": meta["target_param"],
            "policy": meta["policy"],
            "param_value": meta["param_value"],
            "rep": meta["rep"],
            "mean_H": final_cycle_avg,
            "std_H": final_cycle_std,
            "sum_flow_entropy": sum_flow_entropy,
            "sum_quantile_volatility": sum_quantile_volatility,
            "total_performance": total_performance
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
    grid_fig_entropy, grid_ax_entropy = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    grid_fig_quantile_volatility, grid_ax_quantile_volatility = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    grid_fig_performance, grid_ax_performance = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows), layout='constrained')
    
    axes_mean_flat = grid_ax_mean.flatten() if nrows > 1 else np.array([grid_ax_mean]).flatten()
    axes_std_flat = grid_ax_std.flatten() if nrows > 1 else np.array([grid_ax_std]).flatten()
    axes_entropy_flat = grid_ax_entropy.flatten() if nrows > 1 else np.array([grid_ax_entropy]).flatten()
    axes_quantile_volatility_flat = grid_ax_quantile_volatility.flatten() if nrows > 1 else np.array([grid_ax_quantile_volatility]).flatten()
    axes_performance_flat = grid_ax_performance.flatten() if nrows > 1 else np.array([grid_ax_performance]).flatten()

    for i in range(num_params, len(axes_mean_flat)):
        axes_mean_flat[i].set_visible(False)
        axes_std_flat[i].set_visible(False)
        axes_entropy_flat[i].set_visible(False)
        axes_quantile_volatility_flat[i].set_visible(False)
        axes_performance_flat[i].set_visible(False)

    for target_param in active_params:
        param_idx = active_params.index(target_param)
        ax_mean_g = axes_mean_flat[param_idx]
        ax_std_g = axes_std_flat[param_idx]
        ax_entropy_g = axes_entropy_flat[param_idx]
        ax_quantile_volatility_g = axes_quantile_volatility_flat[param_idx]
        ax_performance_g = axes_performance_flat[param_idx]
        
        fig_mean, ax_mean = plt.subplots(figsize=(12, 7), layout='constrained')
        fig_std, ax_std = plt.subplots(figsize=(12, 7), layout='constrained')
        fig_entropy, ax_entropy = plt.subplots(figsize=(12, 7), layout='constrained')
        fig_quantile_volatility, ax_quantile_volatility = plt.subplots(figsize=(12, 7), layout='constrained')
        fig_performance, ax_performance = plt.subplots(figsize=(12, 7), layout='constrained')

        param_df = results_df[results_df["target_param"] == target_param]
        test_values = sorted(param_df["param_value"].unique())
        param_label = param_labels.get(target_param, target_param)
        baseline_val = baseline_parameters.get(target_param)

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
                std_H_q95=("std_H", lambda x: x.quantile(0.95)),
                sum_flow_entropy_avg=("sum_flow_entropy", "mean"),
                sum_flow_entropy_q05=("sum_flow_entropy", lambda x: x.quantile(0.05)),
                sum_flow_entropy_q95=("sum_flow_entropy", lambda x: x.quantile(0.95)),
                sum_quantile_volatility_avg=("sum_quantile_volatility", "mean"),
                sum_quantile_volatility_q05=("sum_quantile_volatility", lambda x: x.quantile(0.05)),
                sum_quantile_volatility_q95=("sum_quantile_volatility", lambda x: x.quantile(0.95)),
                total_performance_avg=("total_performance", "mean"),
                total_performance_q05=("total_performance", lambda x: x.quantile(0.05)),
                total_performance_q95=("total_performance", lambda x: x.quantile(0.95))
            ).reindex(test_values)

            mean_vals = aggregated["mean_H_avg"].values
            mean_q05 = aggregated["mean_H_q05"].values
            mean_q95 = aggregated["mean_H_q95"].values

            std_vals = aggregated["std_H_avg"].values
            std_q05 = aggregated["std_H_q05"].values
            std_q95 = aggregated["std_H_q95"].values

            entropy_vals = aggregated["sum_flow_entropy_avg"].values
            entropy_q05 = aggregated["sum_flow_entropy_q05"].values
            entropy_q95 = aggregated["sum_flow_entropy_q95"].values

            quantile_volatility_vals = aggregated["sum_quantile_volatility_avg"].values
            quantile_volatility_q05 = aggregated["sum_quantile_volatility_q05"].values
            quantile_volatility_q95 = aggregated["sum_quantile_volatility_q95"].values

            performance_vals = aggregated["total_performance_avg"].values
            performance_q05 = aggregated["total_performance_q05"].values
            performance_q95 = aggregated["total_performance_q95"].values

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

            # Plot flow entropy
            ax_entropy.plot(test_values, entropy_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_entropy.errorbar(test_values, entropy_vals, yerr=[entropy_vals - entropy_q05, entropy_q95 - entropy_vals],
                                fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)
            ax_entropy_g.plot(test_values, entropy_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_entropy_g.errorbar(test_values, entropy_vals, yerr=[entropy_vals - entropy_q05, entropy_q95 - entropy_vals],
                                  fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)

            # Plot quantile volatility
            ax_quantile_volatility.plot(test_values, quantile_volatility_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_quantile_volatility.errorbar(test_values, quantile_volatility_vals, yerr=[quantile_volatility_vals - quantile_volatility_q05, quantile_volatility_q95 - quantile_volatility_vals],
                                 fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)
            ax_quantile_volatility_g.plot(test_values, quantile_volatility_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_quantile_volatility_g.errorbar(test_values, quantile_volatility_vals, yerr=[quantile_volatility_vals - quantile_volatility_q05, quantile_volatility_q95 - quantile_volatility_vals],
                                   fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)

            # Plot total performance
            ax_performance.plot(test_values, performance_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_performance.errorbar(test_values, performance_vals, yerr=[performance_vals - performance_q05, performance_q95 - performance_vals],
                                     fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)
            ax_performance_g.plot(test_values, performance_vals, marker='o', label=label, color=color, linewidth=2, alpha=0.8)
            ax_performance_g.errorbar(test_values, performance_vals, yerr=[performance_vals - performance_q05, performance_q95 - performance_vals],
                                       fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.6)

            # Highlight the baseline parameter value with a star if present in swept test_values
            if baseline_val is not None:
                idx_base = None
                for idx_t, val in enumerate(test_values):
                    if np.isclose(val, baseline_val):
                        idx_base = idx_t
                        break
                if idx_base is not None:
                    # Highlight on mean plots
                    ax_mean.plot(test_values[idx_base], mean_vals[idx_base], marker='*', markersize=14,
                                 color=color, markeredgecolor='black', zorder=5)
                    ax_mean_g.plot(test_values[idx_base], mean_vals[idx_base], marker='*', markersize=14,
                                   color=color, markeredgecolor='black', zorder=5)
                    # Highlight on std plots
                    ax_std.plot(test_values[idx_base], std_vals[idx_base], marker='*', markersize=14,
                                color=color, markeredgecolor='black', zorder=5)
                    ax_std_g.plot(test_values[idx_base], std_vals[idx_base], marker='*', markersize=14,
                                  color=color, markeredgecolor='black', zorder=5)
                    # Highlight on flow entropy plots
                    ax_entropy.plot(test_values[idx_base], entropy_vals[idx_base], marker='*', markersize=14,
                                    color=color, markeredgecolor='black', zorder=5)
                    ax_entropy_g.plot(test_values[idx_base], entropy_vals[idx_base], marker='*', markersize=14,
                                      color=color, markeredgecolor='black', zorder=5)
                    # Highlight on quantile volatility plots
                    ax_quantile_volatility.plot(test_values[idx_base], quantile_volatility_vals[idx_base], marker='*', markersize=14,
                                     color=color, markeredgecolor='black', zorder=5)
                    ax_quantile_volatility_g.plot(test_values[idx_base], quantile_volatility_vals[idx_base], marker='*', markersize=14,
                                       color=color, markeredgecolor='black', zorder=5)
                    # Highlight on total performance plots
                    ax_performance.plot(test_values[idx_base], performance_vals[idx_base], marker='*', markersize=14,
                                         color=color, markeredgecolor='black', zorder=5)
                    ax_performance_g.plot(test_values[idx_base], performance_vals[idx_base], marker='*', markersize=14,
                                           color=color, markeredgecolor='black', zorder=5)

        # Finalize individual and grid labels
        for ax, ylabel, title_prefix in [
            (ax_mean, 'Average Health Status', 'Mean'),
            (ax_mean_g, 'Average Health Status', 'Mean'),
            (ax_std, 'Standard Deviation of Health Status', 'Std Dev'),
            (ax_std_g, 'Standard Deviation of Health Status', 'Std Dev'),
            (ax_entropy, 'Sum of Flow Entropy', 'Flow Entropy'),
            (ax_entropy_g, 'Sum of Flow Entropy', 'Flow Entropy'),
            (ax_quantile_volatility, 'Sum of Quantile Volatility', 'Quantile Volatility'),
            (ax_quantile_volatility_g, 'Sum of Quantile Volatility', 'Quantile Volatility'),
            (ax_performance, 'Total Performance', 'Total Performance'),
            (ax_performance_g, 'Total Performance', 'Total Performance')
        ]:
            ax.set_xlabel(param_label, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Save individual plots
        fig_mean.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_mean.png"), dpi=300)
        fig_std.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_std.png"), dpi=300)
        fig_entropy.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_flow_entropy.png"), dpi=300)
        fig_quantile_volatility.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_quantile_volatility.png"), dpi=300)
        fig_performance.savefig(os.path.join(run_dir, f"sensitivity_{target_param}_total_performance.png"), dpi=300)
        plt.close(fig_mean)
        plt.close(fig_std)
        plt.close(fig_entropy)
        plt.close(fig_quantile_volatility)
        plt.close(fig_performance)

    # Save grid plots
    run_dir_name = os.path.basename(os.path.normpath(run_dir))
    grid_fig_mean.suptitle(f"Average H - {run_dir_name}", fontsize=16)
    grid_fig_std.suptitle(f"Standard Deviation of H - {run_dir_name}", fontsize=16)
    grid_fig_entropy.suptitle(f"Flow Entropy - {run_dir_name}", fontsize=16)
    grid_fig_quantile_volatility.suptitle(f"Quantile Volatility - {run_dir_name}", fontsize=16)
    grid_fig_performance.suptitle(f"Total Performance - {run_dir_name}", fontsize=16)

    grid_fig_mean.savefig(os.path.join(run_dir, "grid_sensitivity_mean.png"), dpi=300)
    grid_fig_std.savefig(os.path.join(run_dir, "grid_sensitivity_std.png"), dpi=300)
    grid_fig_entropy.savefig(os.path.join(run_dir, "grid_sensitivity_flow_entropy.png"), dpi=300)
    grid_fig_quantile_volatility.savefig(os.path.join(run_dir, "grid_sensitivity_quantile_volatility.png"), dpi=300)
    grid_fig_performance.savefig(os.path.join(run_dir, "grid_sensitivity_total_performance.png"), dpi=300)
    plt.close(grid_fig_mean)
    plt.close(grid_fig_std)
    plt.close(grid_fig_entropy)
    plt.close(grid_fig_quantile_volatility)
    plt.close(grid_fig_performance)

    print("All plots generated and saved successfully.")
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()
