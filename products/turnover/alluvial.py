import sys
import os
# Add the project root to sys.path to allow running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




# --- LOCAL IMPLEMENTATION OF DATA GENERATION ---
# This avoids importing products.jam.readingGroup which depends on scipy (not installed in exp)
def generate_error_data(params, reps, state_vars, initial_seed, work_dir, engine_path, treatment):
    """Generates simulation data for error bars by running multiple replicates."""
    valid_params = {k: v for k, v in params.items() if v is not False}
    
    seeds = [int(initial_seed + i * 1485) for i in range(reps)]
    sim_configs = []
    for seed in seeds:
        config = {k: [v] for k, v in valid_params.items()}
        config.update({"obsH": [True], "seeds": [seed]})
        sim_configs.append(pd.DataFrame(config))
    
    index_df = pd.concat(sim_configs, ignore_index=True)

    from MyClasses.pathFinder import PathFinder
    producer = PathFinder(working_directory=f'{work_dir}/errorBars/', ENGINE_PATH=engine_path,
                          varsigma=params['varsigma'], OBS_PERIOD=params['OBS_PERIOD'], allRuns_csv=index_df)
    producer.createSelection(name=treatment, minH=0, maxH=1200)
    return producer.produce(selectionName=treatment, stateVariables=state_vars)


# --- SETTINGS ---
SETTINGS = {
    'engine_path': '/Users/Nico/Desktop/CareEngineAnalytics/engine/ABMServer7.jar',
    'base_working_dir': '/Users/Nico/Desktop/simulationOutputs/tryAlluvial/',
    'treatments': ['need', 'risk', 'basal'],
    'reps': 1,
    'state_variables': ['H', 'Delta', 'Disease'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1,
    'csv_filename': 'dominance_lines.csv',
    'selection': 'detail',
    'lambda_to_run': 4.0,
    'bound': 1000/3500, # = 1000 patients for N = 3500, so its the top or bottom 1000

    'subDir': 'prioritization_granularity_1', #
    'fixed_kappa': 0.1,  # change accordingly to subDir

    'varsigma': 300,
    'jaccards': True,
    'vecinity': False,
    'num_deciles': 10,
    'num_observations': 11,
    'start_time': None,
    'end_time': None,
    'color_by_delta': True,
    'prioritization_granularity': 1,
    'orientation': 'horizontal',
}

def process_simulation(params, settings, selection_name, state_vars_override=None):
    """Runs the simulation for a specific configuration and processes the results."""
    params['OBS_PERIOD'] = settings['OBS_PERIOD']
    params['varsigma'] = settings['varsigma']
    params['fixed_kappa'] = settings['fixed_kappa']
    
    gran = settings.get('prioritization_granularity', -1)
    if gran != -1:
        params['prioritization_granularity'] = gran
    elif 'prioritization_granularity' in params:
        params.pop('prioritization_granularity')

    state_vars = state_vars_override if state_vars_override else settings['state_variables']

    raw_data = generate_error_data(
        params=params, 
        reps=settings['reps'], 
        state_vars=state_vars, 
        initial_seed=params['seeds'], 
        work_dir=settings['base_working_dir'], 
        engine_path=settings['engine_path'], 
        treatment=f'{selection_name}/{settings["subDir"]}/{params["Pi"]}'
    )
    return raw_data


def compute_churning(H_df, num_deciles, timesteps):
    """
    Computes the patient quantile churning rate at each transition between observation timesteps.
    Numerator: number of active patients transitioning to a different quantile category.
    Denominator: total number of active patients at both observation times.
    """
    N = len(H_df)
    deciles = {}
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns:
            continue
        h_vals = H_df[col].values
        # Add deterministic tiny noise to make all values unique for qcut
        noise = np.linspace(-1e-12, 1e-12, N)
        sort_idx = np.argsort(h_vals)
        noise_sorted = np.zeros_like(h_vals, dtype=float)
        noise_sorted[sort_idx] = noise
        
        q_labels = pd.qcut(h_vals + noise_sorted, q=num_deciles, labels=False, duplicates='drop')
        deciles[t] = q_labels
        
    churning_rates = []
    for t_idx in range(len(timesteps) - 1):
        t_curr = timesteps[t_idx]
        t_next = timesteps[t_idx + 1]
        
        if t_curr not in deciles or t_next not in deciles:
            continue
            
        dec_curr = deciles[t_curr]
        dec_next = deciles[t_next]
        
        # Check if patients transitioned to a different quantile category
        # A(p) != B(p) is the numerator, all flows in denominator
        valid_mask = (~pd.isna(dec_curr)) & (~pd.isna(dec_next))
        changed_mask = valid_mask & (dec_curr != dec_next)
        
        total = np.sum(valid_mask)
        changed = np.sum(changed_mask)
        
        rate = changed / total if total > 0 else 0.0
        churning_rates.append(rate)
        
    return churning_rates


def compute_flow_entropy(H_df, num_deciles, timesteps):
    """
    Computes the Shannon entropy of the transition flows between consecutive observation timesteps.
    Formula: H = -sum_{A,B} p_{AB} * log2(p_{AB})
    where p_{AB} is the fraction of total patients transitioning from category A to B.
    """
    
    # Theoretical min/max entropy bounds
    #min_entropy = np.log2(num_deciles)
    #max_entropy = 2 * np.log2(num_deciles)

    N = len(H_df)
    deciles = {}
    #Here the deciles are computed:
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns:
            continue
        h_vals = H_df[col].values
        # Add deterministic tiny noise to make all values unique for qcut
        noise = np.linspace(-1e-12, 1e-12, N)
        sort_idx = np.argsort(h_vals)
        noise_sorted = np.zeros_like(h_vals, dtype=float)
        noise_sorted[sort_idx] = noise
        
        q_labels = pd.qcut(h_vals + noise_sorted, q=num_deciles, labels=False, duplicates='drop')
        deciles[t] = q_labels
        
    entropy_values = []
    for t_idx in range(len(timesteps) - 1):
        t_curr = timesteps[t_idx]
        t_next = timesteps[t_idx + 1]
        
        if t_curr not in deciles or t_next not in deciles:
            continue
            
        dec_curr = deciles[t_curr]
        dec_next = deciles[t_next]
        
        # Select active patients at both timesteps
        valid_mask = (~pd.isna(dec_curr)) & (~pd.isna(dec_next))
        curr_valid = dec_curr[valid_mask].astype(int)
        next_valid = dec_next[valid_mask].astype(int)
        
        total = len(curr_valid)
        if total == 0:
            entropy_values.append(0.0)
            continue
            
        # Map transition (A, B) to an integer code: A * num_deciles + B
        transition_codes = curr_valid * num_deciles + next_valid
        
        # Count occurrences of each transition code
        counts = np.bincount(transition_codes, minlength=num_deciles * num_deciles)
        probs = counts / total
        
        # Filter to non-zero probabilities to compute entropy
        active_probs = probs[probs > 0]
        
        # Shannon entropy formula in bits (log base 2)
        entropy = -np.sum(active_probs * np.log2(active_probs))
        entropy_values.append(entropy)
        
    return entropy_values


def compute_kl_divergence(H_df, Delta_df, num_deciles, timesteps):
    """
    Computes the KL divergence (Relative Entropy) of the Severity (Delta) distribution
    across deciles/quantiles at each observation timestep relative to the distribution at time 0.
    """
    try:
        from scipy.stats import entropy
    except ImportError:
        print("Warning: scipy is not installed. Skipping KL divergence calculation.")
        return None
    N = len(H_df)
    deltas = Delta_df['0'].values
    
    # Pre-calculate deciles for all timesteps
    deciles = {}
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns:
            continue
        h_vals = H_df[col].values
        # Add deterministic tiny noise to make all values unique for qcut
        noise = np.linspace(-1e-12, 1e-12, N)
        sort_idx = np.argsort(h_vals)
        noise_sorted = np.zeros_like(h_vals, dtype=float)
        noise_sorted[sort_idx] = noise
        
        q_labels = pd.qcut(h_vals + noise_sorted, q=num_deciles, labels=False, duplicates='drop')
        deciles[t] = q_labels
        
    def get_severity_distribution(t):
        q_labels = deciles[t]
        severity_sums = np.zeros(num_deciles)
        for d in range(num_deciles):
            mask = (q_labels == d)
            if np.any(mask):
                severity_sums[d] = np.nansum(deltas[mask])
        
        total_severity = np.sum(severity_sums)
        if total_severity > 0:
            return severity_sums / total_severity
        else:
            return np.ones(num_deciles) / num_deciles

    # Compute reference distribution qk at time 0 (first timestep)
    t0 = timesteps[0]
    qk = get_severity_distribution(t0)
    
    kl_divs = []
    for t in timesteps:
        if t not in deciles:
            continue
        pk = get_severity_distribution(t)
        # Compute KL divergence using scipy.stats.entropy
        kl = entropy(pk, qk)
        kl_divs.append(kl)
        
    return kl_divs


def compute_severity_entropy(H_df, Delta_df, num_deciles, timesteps):
    """
    Computes the Shannon Entropy (self-information) of the Severity (Delta) distribution
    across deciles/quantiles at each observation timestep.
    For each decile, patient severities (Delta) are binned into 20 discrete bins
    of width 0.5 (from 0 to 10). The Shannon entropy of this 20-bin distribution
    is computed for each decile, and then averaged across all deciles at each timestep.
    """
    try:
        from scipy.stats import entropy
    except ImportError:
        print("Warning: scipy is not installed. Skipping Shannon Entropy of Severity calculation.")
        return None
    N = len(H_df)
    deltas = Delta_df['0'].values
    
    # Pre-calculate deciles for all timesteps
    deciles = {}
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns:
            continue
        h_vals = H_df[col].values
        # Add deterministic tiny noise to make all values unique for qcut
        noise = np.linspace(-1e-12, 1e-12, N)
        sort_idx = np.argsort(h_vals)
        noise_sorted = np.zeros_like(h_vals, dtype=float)
        noise_sorted[sort_idx] = noise
        
        q_labels = pd.qcut(h_vals + noise_sorted, q=num_deciles, labels=False, duplicates='drop')
        deciles[t] = q_labels
        
    avg_entropies = []
    severity_bins = np.linspace(0, 10, 21)  # 20 bins of width 0.5
    
    for t in timesteps:
        if t not in deciles:
            continue
        q_labels = deciles[t]
        decile_entropies = []
        for d in range(num_deciles):
            mask = (q_labels == d)
            if np.any(mask):
                d_deltas = deltas[mask]
                counts, _ = np.histogram(d_deltas, bins=severity_bins)
                total = np.sum(counts)
                if total > 0:
                    pk = counts / total
                    h_d = entropy(pk)
                else:
                    h_d = 0.0
                decile_entropies.append(h_d)
            else:
                decile_entropies.append(0.0)
        avg_entropies.append(np.mean(decile_entropies))
        
    return avg_entropies

def compute_sum_entropy_and_churning(H_df, num_deciles, timesteps):
    """
    Computes the sum of flow entropy and the sum of churning rates across all transition timesteps.
    
    Returns:
        tuple: (sum_flow_entropy, sum_churning)
    """
    churning_rates = compute_churning(H_df, num_deciles, timesteps)
    entropy_values = compute_flow_entropy(H_df, num_deciles, timesteps)
    return sum(entropy_values), sum(churning_rates)


def compute_mutual_information(H_df, Disease_df, timesteps):
    """
    Computes the mutual information (in nats) between H and Disease (severity) at each selected timestep.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        print("Warning: scikit-learn is not installed. Skipping mutual information calculation.")
        return None
    mi_values = []
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns or col not in Disease_df.columns:
            mi_values.append(0.0)
            continue
        
        X = H_df[col].values
        y = Disease_df[col].values
        
        # Filter out any NaNs
        valid_indices = (~np.isnan(X)) & (~np.isnan(y))
        X_clean = X[valid_indices].reshape(-1, 1)
        y_clean = y[valid_indices]
        
        if len(X_clean) > 0:
            mi = mutual_info_regression(X_clean, y_clean, random_state=42)[0]
            mi_values.append(mi)
        else:
            mi_values.append(0.0)
            
    return mi_values


def compute_discretized_mutual_information(H_df, Disease_df, timesteps, bins=10):
    """
    Computes the discretized mutual information using mutual_info_score in nats
    after binning H and Disease (severity) into 10 bins.
    """
    try:
        from sklearn.metrics import mutual_info_score
    except ImportError:
        print("Warning: scikit-learn is not installed. Skipping discretized mutual information calculation.")
        return None
    mi_values = []
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns or col not in Disease_df.columns:
            mi_values.append(0.0)
            continue
        
        X = H_df[col].values
        y = Disease_df[col].values
        
        # Filter out any NaNs
        valid_indices = (~np.isnan(X)) & (~np.isnan(y))
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) > 0:
            # Handle constant arrays where pd.cut fails due to min == max
            if X_clean.max() == X_clean.min():
                X_binned = np.zeros_like(X_clean, dtype=int)
            else:
                X_binned = pd.cut(X_clean, bins=bins, labels=False)
                
            if y_clean.max() == y_clean.min():
                y_binned = np.zeros_like(y_clean, dtype=int)
            else:
                y_binned = pd.cut(y_clean, bins=bins, labels=False)
                
            mi = mutual_info_score(X_binned, y_binned)
            mi_values.append(mi)
        else:
            mi_values.append(0.0)
            
    return mi_values


def compute_normalized_discretized_mutual_information(H_df, Disease_df, timesteps, bins=10):
    """
    Computes the normalized discretized mutual information using normalized_mutual_info_score
    after binning H and Disease (severity) into 10 bins.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        print("Warning: scikit-learn is not installed. Skipping normalized discretized mutual information calculation.")
        return None
    mi_values = []
    for t in timesteps:
        col = str(t)
        if col not in H_df.columns or col not in Disease_df.columns:
            mi_values.append(0.0)
            continue
        
        X = H_df[col].values
        y = Disease_df[col].values
        
        # Filter out any NaNs
        valid_indices = (~np.isnan(X)) & (~np.isnan(y))
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) > 0:
            # Handle constant arrays where pd.cut fails due to min == max
            if X_clean.max() == X_clean.min():
                X_binned = np.zeros_like(X_clean, dtype=int)
            else:
                X_binned = pd.cut(X_clean, bins=bins, labels=False)
                
            if y_clean.max() == y_clean.min():
                y_binned = np.zeros_like(y_clean, dtype=int)
            else:
                y_binned = pd.cut(y_clean, bins=bins, labels=False)
                
            mi = normalized_mutual_info_score(X_binned, y_binned)
            mi_values.append(mi)
        else:
            mi_values.append(0.0)
            
    return mi_values


def plot_alluvial_deciles(H_df, Delta_df, title, save_path, num_deciles=10, num_observations=11, varsigma=300, start_time=None, end_time=None, color_by_delta=False, orientation='horizontal'):
    """
    Plots a clean decile alluvial diagram of H over time, using uniform coloring or Delta coloring.
    """
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = varsigma
        
    timesteps = [int(round(x)) for x in np.linspace(start_time, end_time, num_observations)]
    N = len(H_df)
    
    # 1. Compute H deciles at each selected timestep
    deciles = {}
    for t in timesteps:
        col = str(t)
        h_vals = H_df[col].values
        # Add deterministic tiny noise to make all values unique for qcut
        noise = np.linspace(-1e-12, 1e-12, N)
        sort_idx = np.argsort(h_vals)
        noise_sorted = np.zeros_like(h_vals, dtype=float)
        noise_sorted[sort_idx] = noise
        
        q_labels = pd.qcut(h_vals + noise_sorted, q=num_deciles, labels=False, duplicates='drop')
        deciles[t] = q_labels
    
    # 2. Extract Delta values
    deltas = Delta_df['0'].values
    min_delta = deltas.min()
    max_delta = deltas.max()
    
    def get_color(val):
        if max_delta > min_delta:
            norm = (val - min_delta) / (max_delta - min_delta)
        else:
            norm = 0.0
        return SETTINGS['cmap'](norm)
    
    # 3. Setup vertical layout and styling colors
    slot_height = 1.0 / num_deciles
    decile_height = slot_height * 0.8
    gap = slot_height * 0.2
    
    block_color = '#4a5568'  # Sleek dark slate grey
    flow_color = '#3182ce'   # Beautiful steel blue
    flow_alpha = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 4. Plot decile strata blocks at each timestep
    label_prefix = "D" if num_deciles == 10 else "G"
    avg_deltas = {}
    for t_idx, t in enumerate(timesteps):
        t_deciles = deciles[t]
        avg_deltas[t] = {}
        for d in range(num_deciles):
            pat_indices = np.where(t_deciles == d)[0]
            if len(pat_indices) > 0:
                mean_d = np.nanmean(deltas[pat_indices])
            else:
                mean_d = min_delta
            avg_deltas[t][d] = mean_d
            
            # Draw block
            if color_by_delta:
                rect_color = get_color(mean_d)
            else:
                rect_color = block_color
                
            if orientation == 'horizontal':
                rect = plt.Rectangle((t_idx - 0.15, d * slot_height + gap / 2.0), 0.3, decile_height,
                                     facecolor=rect_color, edgecolor='black', linewidth=1, alpha=0.9, zorder=3)
            else:
                rect = plt.Rectangle((d * slot_height + gap / 2.0, t_idx - 0.15), decile_height, 0.3,
                                     facecolor=rect_color, edgecolor='black', linewidth=1, alpha=0.9, zorder=3)
            ax.add_patch(rect)
            
            if orientation == 'horizontal':
                if t_idx == 0:
                    ax.text(t_idx - 0.2, d * slot_height + slot_height / 2.0, f"{label_prefix}{d+1}", ha='right', va='center', fontsize=10, fontweight='bold', color='black')
                elif t_idx == len(timesteps) - 1:
                    ax.text(t_idx + 0.2, d * slot_height + slot_height / 2.0, f"{label_prefix}{d+1}", ha='left', va='center', fontsize=10, fontweight='bold', color='black')
            else:
                if t_idx == 0:
                    ax.text(d * slot_height + slot_height / 2.0, t_idx - 0.2, f"{label_prefix}{d+1}", ha='center', va='top', fontsize=10, fontweight='bold', color='black')
                elif t_idx == len(timesteps) - 1:
                    ax.text(d * slot_height + slot_height / 2.0, t_idx + 0.2, f"{label_prefix}{d+1}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # 5. Draw alluvial flows (ribbons) between adjacent timesteps
    num_steps_between = 30
    x_steps = np.linspace(0, 1, num_steps_between)
    smooth_t = 3 * x_steps**2 - 2 * x_steps**3
    
    for t_idx in range(len(timesteps) - 1):
        t_curr = timesteps[t_idx]
        t_next = timesteps[t_idx + 1]
        
        dec_curr = deciles[t_curr]
        dec_next = deciles[t_next]
        
        flow_pats = {A: {B: [] for B in range(num_deciles)} for A in range(num_deciles)}
        for p in range(N):
            A = dec_curr[p]
            B = dec_next[p]
            if pd.isna(A) or pd.isna(B):
                continue  # skip patients with NaN values (e.g. dropped out)
            A = int(A)
            B = int(B)
            flow_pats[A][B].append(p)
            
        outgoing_offsets = {A: 0.0 for A in range(num_deciles)}
        incoming_offsets = {B: 0.0 for B in range(num_deciles)}
        
        y_starts = {A: {} for A in range(num_deciles)}
        y_ends = {A: {} for A in range(num_deciles)}
        
        # Compute outgoing positions
        for A in range(num_deciles):
            total_in_A = sum(len(flow_pats[A][B]) for B in range(num_deciles))
            if total_in_A == 0:
                continue
            for B in range(num_deciles):
                cnt = len(flow_pats[A][B])
                if cnt == 0:
                    continue
                h_flow = (cnt / total_in_A) * decile_height
                y_bottom = A * slot_height + gap / 2.0 + outgoing_offsets[A]
                y_top = y_bottom + h_flow
                y_starts[A][B] = (y_top, y_bottom)
                outgoing_offsets[A] += h_flow
                
        # Compute incoming positions
        for B in range(num_deciles):
            total_in_B = sum(len(flow_pats[A][B]) for A in range(num_deciles))
            if total_in_B == 0:
                continue
            for A in range(num_deciles):
                cnt = len(flow_pats[A][B])
                if cnt == 0:
                    continue
                h_flow = (cnt / total_in_B) * decile_height
                y_bottom = B * slot_height + gap / 2.0 + incoming_offsets[B]
                y_top = y_bottom + h_flow
                y_ends[A][B] = (y_top, y_bottom)
                incoming_offsets[B] += h_flow
                
        # Draw the ribbons
        for A in range(num_deciles):
            for B in range(num_deciles):
                pats = flow_pats[A][B]
                if len(pats) == 0:
                    continue
                
                # Determine ribbon color
                if color_by_delta:
                    # ribbon from A to B is colored with average delta of patients on B at next step
                    mean_delta = avg_deltas[t_next][B]
                    color = get_color(mean_delta)
                else:
                    color = flow_color
                
                y_start_top, y_start_bottom = y_starts[A][B]
                y_end_top, y_end_bottom = y_ends[A][B]
                
                if orientation == 'horizontal':
                    x1, x2 = t_idx, t_idx + 1
                    seg_x = np.linspace(x1, x2, num_steps_between)
                    
                    curve_top = y_start_top + (y_end_top - y_start_top) * smooth_t
                    curve_bottom = y_start_bottom + (y_end_bottom - y_start_bottom) * smooth_t
                    
                    ax.fill_between(seg_x, curve_bottom, curve_top, color=color, alpha=flow_alpha, zorder=2)
                else:
                    y1, y2 = t_idx, t_idx + 1
                    seg_y = np.linspace(y1, y2, num_steps_between)
                    
                    curve_right = y_start_top + (y_end_top - y_start_top) * smooth_t
                    curve_left = y_start_bottom + (y_end_bottom - y_start_bottom) * smooth_t
                    
                    ax.fill_betweenx(seg_y, curve_left, curve_right, color=color, alpha=flow_alpha, zorder=2)

    # Styling and Labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    decile_label = 'Deciles of Health Problems' if num_deciles == 10 else f'{num_deciles} Quantiles of Health Problems'
    
    if orientation == 'horizontal':
        ax.set_xlabel('Time (Simulation Cycles)', fontsize=12, labelpad=10)
        ax.set_ylabel(decile_label, fontsize=12, labelpad=10)
        
        ax.set_xticks(range(len(timesteps)))
        ax.set_xticklabels(timesteps)
        ax.set_xlim(-0.5, len(timesteps) - 0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([])
    else:
        ax.set_ylabel('Time (Simulation Cycles)', fontsize=12, labelpad=10)
        ax.set_xlabel(decile_label, fontsize=12, labelpad=10)
        
        ax.set_yticks(range(len(timesteps)))
        ax.set_yticklabels(timesteps)
        ax.set_ylim(-0.5, len(timesteps) - 0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([])
        
    ax.grid(False)
    
    if color_by_delta:
        # Add colorbar for Delta values
        cmap = SETTINGS['cmap']
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_delta, vmax=max_delta))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        cbar.set_label('Number of Diseases', fontsize=12, labelpad=10)
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Alluvial plot saved to {save_path}")


def main():
    # --- Path Handling for multiple machines ---
    current_base_working_dir = SETTINGS['base_working_dir']
    current_engine_path = SETTINGS['engine_path']

    # Check if the current base_working_dir exists
    if not os.path.exists(current_base_working_dir):
        print(f"Warning: Base working directory '{current_base_working_dir}' not found. Trying alternative path.")
        # Assume the other user's path
        if "/Users/Nico" in current_base_working_dir:
            SETTINGS['base_working_dir'] = current_base_working_dir.replace("/Users/Nico", "/Users/nicolasbarticevic")
            SETTINGS['engine_path'] = current_engine_path.replace("/Users/Nico", "/Users/nicolasbarticevic")
        elif "/Users/nicolasbarticevic" in current_base_working_dir:
            SETTINGS['base_working_dir'] = current_base_working_dir.replace("/Users/nicolasbarticevic", "/Users/Nico")
            SETTINGS['engine_path'] = current_engine_path.replace("/Users/nicolasbarticevic", "/Users/Nico")
        else:
            print("Error: Neither '/Users/Nico' nor '/Users/nicolasbarticevic' found in base_working_dir. Please check paths.")
            return # Exit if paths are completely unexpected

        print(f"Updated base working directory to: {SETTINGS['base_working_dir']}")
        print(f"Updated engine path to: {SETTINGS['engine_path']}")

    # 1. Load Configuration Data
    data_path = os.path.join(SETTINGS['base_working_dir'], SETTINGS['csv_filename'])
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find configuration file at {data_path}")
        return

    # 2. Run Simulations and Collect Data
    mech_data = {}
    selection = SETTINGS['selection']
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

    # 3. Generate Alluvial Diagrams for each policy
    for schedule, raw_data in mech_data.items():
        # extract the single replicate's DataFrame for 'H' and 'Delta'
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        Delta_df = rep_data['Delta']
        
        output_path = os.path.join(SETTINGS['base_working_dir'], f"alluvial_{schedule}.png")
        num_deciles = SETTINGS.get('num_deciles', 10)
        num_observations = SETTINGS.get('num_observations', 10)
        varsigma = SETTINGS.get('varsigma', 300)
        start_time = SETTINGS.get('start_time', None)
        end_time = SETTINGS.get('end_time', None)
        color_by_delta = SETTINGS.get('color_by_delta', False)
        
        policy_labels = {"need": "Need Prioritisation", "risk": "Risk Stratification", "basal": "FCFS"}
        label = policy_labels.get(schedule, schedule.upper())
        label_prefix = "Deciles" if num_deciles == 10 else f"{num_deciles}-Quantiles"
        color_suffix = " (Colored by Average Number of Diseases)" if color_by_delta else ""
        title = f"{label} - Health {label_prefix} Alluvial Diagram{color_suffix}"
        orientation = SETTINGS.get('orientation', 'horizontal')
        plot_alluvial_deciles(H_df, Delta_df, title, output_path, num_deciles=num_deciles,
                              num_observations=num_observations, varsigma=varsigma,
                              start_time=start_time, end_time=end_time,
                              color_by_delta=color_by_delta, orientation=orientation)

    # 4. Compute Churning Rates and Plot Churning Rate Over Time
    churning_results = {}
    
    num_deciles = SETTINGS.get('num_deciles', 10)
    num_observations = SETTINGS.get('num_observations', 10)
    varsigma = SETTINGS.get('varsigma', 300)
    start_time = SETTINGS.get('start_time', None)
    end_time = SETTINGS.get('end_time', None)
    
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = varsigma
        
    timesteps = [int(round(x)) for x in np.linspace(start_time, end_time, num_observations)]
    
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        
        rates = compute_churning(H_df, num_deciles, timesteps)
        churning_results[schedule] = rates
        
    if churning_results:
        transition_times = timesteps[1:]
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in churning_results:
                continue
            rates = churning_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = transition_times[:len(rates)]
            plt.plot(x_vals, rates, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        label_prefix = "Decile" if num_deciles == 10 else f"{num_deciles}-Quantile"
        plt.title(f"Patient Health {label_prefix} Churning Rate Over Time", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle (Transition End Time)", fontsize=12, labelpad=10)
        plt.ylabel("Churning Rate (Proportion Changing Quantile)", fontsize=12, labelpad=10)
        plt.ylim(-0.05, 1.05)
        plt.xticks(transition_times)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=11)
        plt.tight_layout()
        
        churn_plot_path = os.path.join(SETTINGS['base_working_dir'], "churning_rate.png")
        plt.savefig(churn_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Churning rate plot saved to {churn_plot_path}")

    # 5. Compute Shannon Entropy of Transition Flows and Plot Over Time
    entropy_results = {}
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        
        entropies = compute_flow_entropy(H_df, num_deciles, timesteps)
        entropy_results[schedule] = entropies
        
    if entropy_results:
        transition_times = timesteps[1:]
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in entropy_results:
                continue
            entropies = entropy_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = transition_times[:len(entropies)]
            plt.plot(x_vals, entropies, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        label_prefix = "Decile" if num_deciles == 10 else f"{num_deciles}-Quantile"
        
        # Reference lines for theoretical min/max entropy bounds
        min_entropy = np.log2(num_deciles)
        max_entropy = 2 * np.log2(num_deciles)
        
        plt.axhline(y=min_entropy, color='#cbd5e0', linestyle='--', linewidth=1.5, label='Min Theoretical Entropy (No Churn)')
        plt.axhline(y=max_entropy, color='#feb2b2', linestyle='--', linewidth=1.5, label='Max Theoretical Entropy (Full Shuffling)')
        
        plt.title(f"Transition Flow Shannon Entropy Over Time ({label_prefix}s)", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle (Transition End Time)", fontsize=12, labelpad=10)
        plt.ylabel("Shannon Entropy (Bits)", fontsize=12, labelpad=10)
        plt.ylim(min_entropy - 0.2, max_entropy + 0.2)
        plt.xticks(transition_times)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        entropy_plot_path = os.path.join(SETTINGS['base_working_dir'], "flow_entropy.png")
        plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Flow entropy plot saved to {entropy_plot_path}")

    # 6. Compute KSG Mutual Information between H and Severity (Disease) and Plot Over Time
    mi_results = {}
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        Disease_df = rep_data.get('Disease')
        
        if Disease_df is not None and not Disease_df.empty:
            print(f"Computing KSG mutual information for schedule: {schedule}...")
            mi_values = compute_mutual_information(H_df, Disease_df, timesteps)
            if mi_values is not None:
                mi_results[schedule] = mi_values
        else:
            print(f"Warning: 'Disease' data not available for schedule '{schedule}', skipping KSG mutual information calculation.")
            
    if mi_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in mi_results:
                continue
            mi_values = mi_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = timesteps[:len(mi_values)]
            plt.plot(x_vals, mi_values, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        plt.title(r"KSG Mutual Information between Health Status (H) and Severity ($\delta$) over Time", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("KSG Mutual Information (Nats)", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        mi_plot_path = os.path.join(SETTINGS['base_working_dir'], "ksg_mutual_information_H_severity.png")
        plt.savefig(mi_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"KSG mutual information plot saved to {mi_plot_path}")

    # 7. Compute Discretized Mutual Information between H and Severity (Disease) and Plot Over Time
    disc_mi_results = {}
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        Disease_df = rep_data.get('Disease')
        
        if Disease_df is not None and not Disease_df.empty:
            print(f"Computing discretized mutual information for schedule: {schedule}...")
            disc_mi_values = compute_discretized_mutual_information(H_df, Disease_df, timesteps, bins=10)
            if disc_mi_values is not None:
                disc_mi_results[schedule] = disc_mi_values
        else:
            print(f"Warning: 'Disease' data not available for schedule '{schedule}', skipping discretized mutual information calculation.")
            
    if disc_mi_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in disc_mi_results:
                continue
            disc_mi_values = disc_mi_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = timesteps[:len(disc_mi_values)]
            plt.plot(x_vals, disc_mi_values, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        plt.title(r"Discretized Mutual Information between Health Status (H) and Severity ($\delta$) over Time (10 Bins)", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("Discretized Mutual Information (Nats)", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        disc_mi_plot_path = os.path.join(SETTINGS['base_working_dir'], "discretized_mutual_information_H_severity.png")
        plt.savefig(disc_mi_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Discretized mutual information plot saved to {disc_mi_plot_path}")

    # 8. Compute Normalized Discretized Mutual Information between H and Severity (Disease) and Plot Over Time
    norm_mi_results = {}
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        Disease_df = rep_data.get('Disease')
        
        if Disease_df is not None and not Disease_df.empty:
            print(f"Computing normalized discretized mutual information for schedule: {schedule}...")
            norm_mi_values = compute_normalized_discretized_mutual_information(H_df, Disease_df, timesteps, bins=10)
            if norm_mi_values is not None:
                norm_mi_results[schedule] = norm_mi_values
        else:
            print(f"Warning: 'Disease' data not available for schedule '{schedule}', skipping normalized discretized mutual information calculation.")
            
    if norm_mi_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in norm_mi_results:
                continue
            norm_mi_values = norm_mi_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = timesteps[:len(norm_mi_values)]
            plt.plot(x_vals, norm_mi_values, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        plt.title(r"Normalized Discretized Mutual Information between Health Status (H) and Severity ($\delta$) over Time (10 Bins)", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("Normalized Discretized Mutual Information", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        norm_mi_plot_path = os.path.join(SETTINGS['base_working_dir'], "normalized_discretized_mutual_information_H_severity.png")
        plt.savefig(norm_mi_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normalized discretized mutual information plot saved to {norm_mi_plot_path}")

    # 9. Compute Relative Entropy (KL Divergence) of Severity (Delta) across Deciles and Plot Over Time
    kl_results = {}
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        Delta_df = rep_data['Delta']
        
        print(f"Computing Relative Entropy (KL divergence) for schedule: {schedule}...")
        kl_values = compute_kl_divergence(H_df, Delta_df, num_deciles, timesteps)
        if kl_values is not None:
            kl_results[schedule] = kl_values
        
    if kl_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in kl_results:
                continue
            kl_values = kl_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = timesteps[:len(kl_values)]
            plt.plot(x_vals, kl_values, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        label_prefix = "Decile" if num_deciles == 10 else f"{num_deciles}-Quantile"
        plt.title(f"Relative Entropy (KL Divergence) of Severity across {label_prefix}s over Time", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("KL Divergence from Time 0 (Nats)", fontsize=12, labelpad=10)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        kl_plot_path = os.path.join(SETTINGS['base_working_dir'], "kl_divergence_severity.png")
        plt.savefig(kl_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"KL divergence plot saved to {kl_plot_path}")

    # 10. Compute Shannon Entropy (self-information) of Severity (Delta) across Deciles and Plot Over Time
    entropy_severity_results = {}
    for schedule, raw_data in mech_data.items():
        rep_data = list(raw_data.values())[0]
        H_df = rep_data['H']
        Delta_df = rep_data['Delta']
        
        print(f"Computing Shannon Entropy of Severity for schedule: {schedule}...")
        entropy_values = compute_severity_entropy(H_df, Delta_df, num_deciles, timesteps)
        if entropy_values is not None:
            entropy_severity_results[schedule] = entropy_values
        
    if entropy_severity_results:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        style_map = {
            'need': {'color': '#E53E3E', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'label': 'Need-based'},
            'risk': {'color': '#DD6B20', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'Risk-based'},
            'basal': {'color': '#4A5568', 'marker': '^', 'linestyle': ':', 'linewidth': 2, 'label': 'Basal (FCFS)'}
        }
        
        for schedule in ['basal', 'risk', 'need']:  # Order them logically in legend
            if schedule not in entropy_severity_results:
                continue
            entropy_values = entropy_severity_results[schedule]
            style = style_map.get(schedule, {'color': '#718096', 'marker': 'x', 'linestyle': '-', 'linewidth': 2, 'label': schedule})
            
            x_vals = timesteps[:len(entropy_values)]
            plt.plot(x_vals, entropy_values, label=style['label'], color=style['color'], 
                     marker=style['marker'], linestyle=style['linestyle'], linewidth=style['linewidth'],
                     markersize=8, alpha=0.9)
            
        label_prefix = "Decile" if num_deciles == 10 else f"{num_deciles}-Quantile"
        
        # Max theoretical discrete severity entropy is log(20) because of 20 severity bins
        max_entropy = np.log(20)
        plt.axhline(y=max_entropy, color='#feb2b2', linestyle='--', linewidth=1.5, label='Max Theoretical Entropy (Uniform)')
        
        plt.title(f"Shannon Entropy (Self-Information) of Severity across {label_prefix}s over Time", 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Simulation Cycle", fontsize=12, labelpad=10)
        plt.ylabel("Shannon Entropy (Nats)", fontsize=12, labelpad=10)
        plt.ylim(0.0, max_entropy + 0.1)
        plt.xticks(timesteps)
        plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
        plt.tight_layout()
        
        entropy_plot_path = os.path.join(SETTINGS['base_working_dir'], "severity_entropy.png")
        plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Shannon Entropy (self-information) of Severity plot saved to {entropy_plot_path}")
if __name__ == "__main__":
    main()
