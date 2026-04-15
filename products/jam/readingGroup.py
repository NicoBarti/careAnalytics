import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from MyClasses.pathFinder import PathFinder
from MyClasses.trajectoryDistances import TrajectoryDistances
from MyClasses.linePlotter import LinePlotter

# --- CONFIGURATION ---
FIG_NAME = "superLond100000"
SETTINGS = {
    'engine_path': '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar',
    'base_working_dir': '/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/',
    #'treatments': [f'{FIG_NAME}/basal', f'{FIG_NAME}/risk', f'{FIG_NAME}/need'],
    'treatments': [ f'{FIG_NAME}/need'],
    'reps': 1,
    'initial_seed': 52920828945,
    'fixed_seed': 62920828945,
    'state_variables': ['H', 'N', 'SimpleC', 'SimpleB', 'T', 'Disease', 'Performance'],
    #'state_variables': ['H','T'],
    'cmap': mpl.colormaps['plasma'],
    'OBS_PERIOD': 1000,
    'run_jaccard_analysis': False,
    'plot_seeking_on_twinx': False,
    'maxy': 10
}

# --- SIMULATION & DATA GENERATION ---

def generate_error_data(params, reps, state_vars, initial_seed, work_dir, engine_path, treatment):
    """Generates simulation data for error bars by running multiple replicates."""
    # Filter out False parameters
    valid_params = {k: v for k, v in params.items() if v != False}
    
    # Create simulation index
    seeds = [int(initial_seed + i * 1485) for i in range(reps)]
    sim_configs = []
    for seed in seeds:
        config = {k: [v] for k, v in valid_params.items()}
        config.update({"obsH": [True], "seeds": [seed], "H": [8]})
        sim_configs.append(pd.DataFrame(config))
    
    index_df = pd.concat(sim_configs, ignore_index=True)

    producer = PathFinder(working_directory=f'{work_dir}/errorBars/', ENGINE_PATH=engine_path,
                          varsigma=params['varsigma'], OBS_PERIOD=params['OBS_PERIOD'], allRuns_csv=index_df)
    producer.createSelection(name=treatment, minH=0, maxH=1200)
    return producer.produce(selectionName=treatment, stateVariables=state_vars)

# --- ANALYSIS HELPER FUNCTIONS ---

def compute_jaccard(data_dict, state_var, all_replicates=False):
    """Computes Jaccard similarity matrix for a state variable.
    
    Args:
        data_dict (dict): Dictionary of simulation runs.
        state_var (str): The state variable to analyze.
        all_replicates (bool): If True, averages over all replicates. If False, uses only the first.
    """
    if not all_replicates:
        # Original behavior: use only the first replicate
        simulation = data_dict[next(iter(data_dict))]
        num_windows = simulation[state_var].shape[1]
        jaccardMatrix = np.empty((num_windows, num_windows))
        data = simulation[state_var].values > 0
        
        for i in range(num_windows):
            for j in range(num_windows):
                union = data[:, i] | data[:, j]
                intersection = data[:, i] & data[:, j]
                if not union.any():
                    jaccardMatrix[i, j] = np.nan
                else:
                    jaccardMatrix[i, j] = intersection.sum() / union.sum()
        return jaccardMatrix
    
    # Average over all replicates
    all_matrices = []
    for simulation in data_dict.values():
        num_windows = simulation[state_var].shape[1]
        jaccard_matrix = np.empty((num_windows, num_windows))
        data = simulation[state_var].values > 0
        
        for i in range(num_windows):
            for j in range(num_windows):
                union = data[:, i] | data[:, j]
                intersection = data[:, i] & data[:, j]
                if not union.any():
                    jaccard_matrix[i, j] = np.nan
                else:
                    jaccard_matrix[i, j] = intersection.sum() / union.sum()
        all_matrices.append(jaccard_matrix)
    
    return all_matrices

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

def compute_correlations_and_percentages(error_data, capacity):
    """Computes correlations between Need/Use and usage percentages."""
    need_arr = np.array([d['N'] for d in error_data.values()])
    use_arr = np.array([d['SimpleC'] for d in error_data.values()])
    
    corrs, percents = [], []
    for e in range(need_arr.shape[0]):
        # Per simulation
        sim_n, sim_u = need_arr[e], use_arr[e]
        
        # Correlations per time window
        c_list = [np.corrcoef(sim_n[:, i], sim_u[:, i])[0, 1] for i in range(sim_n.shape[1])]
        corrs.append(c_list)
        
        # Percentages per time window
        p_list = []
        for i in range(sim_n.shape[1]):
            # Top 'capacity' needs
            top_indices = np.argpartition(-sim_n[:, i], capacity)[:capacity]
            p_list.append(sim_u[:, i].take(top_indices).sum() / capacity)
        percents.append(p_list)
        
    return np.array(corrs), np.array(percents)

# --- PLOTTING ---

def get_style(treatment):
    styles = {
        'basal': (0, "First-Come, First-Served", "dotted"),
        'risk': (1, "Risk-Stratification", "dashed"),
        'need': (2, "Need-Prioritization", "solid")
    }
    key = next((k for k in styles if k in treatment), None)
    return styles.get(key, (0, treatment, "solid"))

def plot_histogram(ax, error_data, title, color, var = 'H', xlab = "Unserved Needs", ylab = "Patients",
                   hist_label = None):
    """Plots histogram of final H values."""
    final_h = np.array([d[var].iloc[:, -1] for d in error_data.values()])

    # Compute stats
    counts_list = [np.histogram(row, bins=20)[0] for row in final_h]
    mean_counts = np.mean(counts_list, axis=0)
    quantiles = np.quantile(counts_list, q=[0.05, 0.95], axis=0)
    edges = np.histogram(final_h[0], bins = 20)[1]

    # Plot
    midbin=(edges[1] - edges[0])/2
    ax.stairs(mean_counts, edges, fill=True, color=color, label = hist_label)
    ax.errorbar(edges[:-1]+midbin, mean_counts, yerr=np.abs(quantiles - mean_counts), fmt='.', color='black')

#    ax.errorbar(np.arange(0.5, 19.5), mean_counts, yerr=np.abs(quantiles - mean_counts), fmt='.', color='black')
    
    #ax.set_ylim(top=1200)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)

def plot_temporal_series(ax, data, label, color, linestyle, y_label, title=None, maxy=None):
    """Plots temporal series with error bars."""
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

def plot_jaccard_analysis(jaccard_data, seeker_data, treatments=None, optionalTitle = None):
    """Plots heatmaps for Jaccard and Vicinity analysis."""
    if treatments is None:
        treatments = SETTINGS['treatments']

    metrics = [
        ('SimpleB', 'B', "Jaccard Seek-Care", "Seeking Similarity"),
        ('T', 'T', "Jaccard Treatment", "Treatment Similarity"),
        ('sN', 'sN', "Seeker Needs", "Avg Needs (Seekers)"),
        ('tN', 'tN', "Treated Needs", "Avg Needs (Treated)")
    ]
    
    data_source = {'B': jaccard_data['SimpleB'], 'T': jaccard_data['T'], 
                   'sN': seeker_data['sN'], 'tN': seeker_data['tN']}

    returnPlots = []
    for key, code, title, subtitle in metrics:
        c = mpl.colormaps['summer']
        for treatment in treatments:
            _, style_title, _ = get_style(treatment)
            matrix = data_source[code].get(treatment)
            if matrix is None: continue

            fig, (ax_heat, ax_diag) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            
            sns.heatmap(matrix, ax=ax_heat)
            ax_heat.set_title(subtitle)
            
            ax_diag.plot(matrix.diagonal(), color=c(1), label='t', linestyle='solid')
            if matrix.shape[0] > 1: ax_diag.plot(matrix.diagonal(offset=1), color=c(0.4), label='t+1', linestyle='solid')
            #if matrix.shape[0] > 2: ax_diag.plot(matrix.diagonal(offset=2), color=c(0.7), label='t+2', linestyle='solid')
            if matrix.shape[0] > 1: ax_diag.plot(matrix.diagonal(offset=-1), color=c(0.7), label='t-1', linestyle='dashed')
            ax_diag.set_title("Diagonal Decay")
            ax_diag.legend()
            
            fig.suptitle(f"{title} - {style_title} - {optionalTitle}")
            plt.show()
            returnPlots.append(fig)
    return returnPlots


# --- MAIN ---

def main():
    # 1. Setup Figures
    fig_hist, ax_hist = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig_main, ax_main = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_twin = ax_main[1].twinx() if SETTINGS['plot_seeking_on_twinx'] else None
    fig_corr, ax_corr = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    
    # Store Analysis Data
    jaccard_results = {'SimpleB': {}, 'T': {}}
    seeker_results = {'sN': {}, 'tN': {}}

    # 2. Process Treatments
    for treatment in SETTINGS['treatments']:
        print(f"Processing: {treatment}")
        
        # Load params & Setup
        idx, title, style = get_style(treatment)
        color = SETTINGS['cmap'](0.6 - (idx * 0.3))
        
        df_run = pd.read_csv(f"{SETTINGS['base_working_dir']}/{treatment}.csv")
        dist_tool = TrajectoryDistances(working_directory=SETTINGS['base_working_dir'], allRuns_csv=df_run,
                                        ENGINE_PATH=SETTINGS['engine_path'], varsigma=300, oederByWindow = 0,
                                        selectionName=treatment, OBS_PERIOD=SETTINGS['OBS_PERIOD'], minH = 0,
                                        maxH = 99999999999999999999, orderByVariable = 'H')
        params = dist_tool.get_seedParams(seed=df_run['seeds'].iloc[0], selectionName=treatment, filterParams=False)

        # Generate Data
        err_data = generate_error_data(params, SETTINGS['reps'], SETTINGS['state_variables'], 
                                       SETTINGS['initial_seed'], SETTINGS['base_working_dir'], 
                                       SETTINGS['engine_path'], treatment)

        # Plot Histogram
        plot_histogram(ax_hist[idx], err_data, title, color)

        # Extract Arrays for Series
        N_arr = np.array([d['N'].mean(0) for d in err_data.values()])
        B_arr = np.array([d['SimpleB'].mean(0) for d in err_data.values()])
        T_arr = np.array([d['T'].replace(0, np.nan).mean(0) for d in err_data.values()]) # Handle 0s as NaNs

        # Plot Main Figures (Performance & Needs)
        plot_temporal_series(ax_main[0], T_arr, title, color, style, "Needs Solved/Visit", "Performance")
        plot_temporal_series(ax_main[1], N_arr, f"Needs ({title.split()[0]})", color, 'solid', 
                    "Unsolved Needs", "Population Status", maxy=SETTINGS['maxy'])
        
        if ax_twin:
            plot_temporal_series(ax_twin, B_arr, f"Seeking ({title.split()[0]})", color, 'dotted', "Proportion Seeking")

        # Jaccard Analysis
        if SETTINGS['run_jaccard_analysis']:
            jaccard_results['SimpleB'][treatment] = compute_jaccard(err_data, 'SimpleB')
            jaccard_results['T'][treatment] = compute_jaccard(err_data, 'T')
            seeker_results['sN'][treatment] = compute_vicinity(err_data, 'N', "seekers")
            seeker_results['tN'][treatment] = compute_vicinity(err_data, 'N', "treated")

        # Correlation Analysis
        corrs, percents = compute_correlations_and_percentages(err_data, params['totalCapacity'])
        plot_temporal_series(ax_corr[0], corrs, title, color, style, "Correlation", "Needs vs Use Correlation")
        plot_temporal_series(ax_corr[1], percents, title, color, style, "Percentage", "Needs vs Use Percentage")

    # 3. Finalize
    ax_main[1].set_ylim(bottom=0)
    fig_hist.suptitle("Unserved Needs Distribution", size=14)
    fig_main.suptitle("System Performance & Behaviour", size=14)
    
    plt.show()

    if SETTINGS['run_jaccard_analysis']:
        plot_jaccard_analysis(jaccard_results, seeker_results)

if __name__ == "__main__":
    main()
