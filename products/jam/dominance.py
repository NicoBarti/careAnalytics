from MyClasses.client import Client
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

HOME_DIR = Path.home()
ENGINE_PATH = HOME_DIR / 'Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
BASE_WORKING_DIRECTORY = HOME_DIR / 'Desktop/simulationOutputs/JAMPaper/dominance/'

BASE_PARAMS = {
    # Fixed parameters
    'varsigma': [100],
    'OBS_PERIOD': [100],  # only interested in the end value
    'N': [3400],
    'totalCapacity': [170],
    'random_delta_min': [0],
    'random_delta_max': [10],
    'fixed_tau': [2],
    'fixed_capN': [10],
    'fixed_capE': [10],
    'fixed_kappa': [0.5],
    # Common parameters
    'Pi': ['basal'],
    'reproduce_line': ['true'],
    'obsH': ['true'],
    'obsT': ['true']
}

EXPLORATION_SPECIFIC_PARAMS = {
    "middle_ground": {
        'W': [25],
        'fixed_lambda': [4],
        'fixed_rho': [7],
        'fixed_eta': [8],
        'fixed_psi': [0.5],
    },
    "need_ground": {
        'W': [35],
        'fixed_lambda': [6],
        'fixed_rho': [9],
        'fixed_eta': [9],
        'fixed_psi': [0.9],
    },
    "risk_ground": {
        'W': [11],
        'fixed_lambda': [1],
        'fixed_rho': [5],
        'fixed_eta': [7],
        'fixed_psi': [0.1],
    }

}


def get_Params(exploration: str):
    """A base param grid with fixed values, updated by exploration type."""
    if exploration not in EXPLORATION_SPECIFIC_PARAMS:
        raise ValueError(f"Unknown exploration type: {exploration}")

    params = BASE_PARAMS.copy()
    params.update(EXPLORATION_SPECIFIC_PARAMS[exploration])
    return params


def run_simulation_scan(client, base_params, param_name, start, stop, num, base_working_directory, search_name):
    save_path = Path(base_working_directory) / search_name / param_name
    save_path.mkdir(parents=True, exist_ok=True)
    param_values = np.linspace(start=start, stop=stop, num=num)

    for policy in ['risk', 'need']:
        Hs, Ts = np.empty((0, base_params['N'][0])), np.empty((0, base_params['N'][0]))
        for val in param_values:
            base_params[param_name] = [val]
            base_params['Pi'] = [policy]
            base_params['W'] = [int(base_params['W'][0])]
            print(f'{policy} {param_name} {val}')
            run_data = client.socket_with_model_paramGrid_2(gridParameters=pd.DataFrame(base_params))
            try:
                client.start_server()
            except ConnectionRefusedError as e:
                print(f"Parece que Java refused: {e}")
                print(f"Tratando de nuevo")
                time.sleep(2)
                client.start_server()
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

            Hs = np.concatenate([Hs, [np.array(run_data[0]['H'])[:, 1]]], axis=0)
            Ts = np.concatenate([Ts, [np.array(run_data[0]['T'])[:, 1]]], axis=0)

        file_name_base = f'{policy}_start{start}_stop{stop}_num{num}'
        np.save(save_path / f'{file_name_base}_Hs.npy', arr=Hs)
        np.save(save_path / f'{file_name_base}_Ts.npy', arr=Ts)


def load_simulation_data(base_working_directory, search_name, param_name, policy, start, stop, num):
    """
    Loads the .npy files for a given simulation configuration.

    Returns:
        tuple: (Hs, Ts) arrays loaded from the files.
    """
    load_path = Path(base_working_directory) / search_name / param_name
    file_name_base = f'{policy}_start{start}_stop{stop}_num{num}'
    hs_file = load_path / f'{file_name_base}_Hs.npy'
    ts_file = load_path / f'{file_name_base}_Ts.npy'

    if not hs_file.exists() or not ts_file.exists():
        raise FileNotFoundError(f"Data files not found in {load_path}")

    Hs = np.load(hs_file)
    Ts = np.load(ts_file)
    Ts[Ts == 0] = np.nan
    return Hs, Ts


def combinedBars(results, search, title, metric, exploration):
    # global color norm:
    dominances = {}
    for param, values in results.items():
        dominances[param] = values['need'] - values['risk']
    norm = mcolors.CenteredNorm(halfrange=np.nanmax(np.abs(np.array(list(dominances.values())))))
    # reference base parameters
    ref_params = get_Params(exploration)

    # a horizontal grid of param columns
    num_params = len(results.keys())
    fig, axe = plt.subplots(figsize=(9.2, 5), ncols=1, nrows=num_params, layout="constrained", squeeze=False)
    for ax, par in zip(axe.flatten(), results.keys()):
        ax.set_xlim(left=search[par]['start'], right=search[par]['stop'])
        width = (search[par]['stop'] - search[par]['start']) / search[par]['num']
        ax.set_xlabel(par)
        for segment, dom in zip(np.linspace(start=search[par]['start'], stop=search[par]['stop'],
                                            num=search[par]['num']), dominances[par]):
            ax.barh(par, width, left=segment, height=0.5, color=plt.get_cmap('PRGn')(norm(dom)))
        # add reference param valua (base param)
        ax.vlines(ref_params[par][0], 0, 1)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axe.ravel().tolist())
    cbar.set_label(f'Difference in {metric} between Strategies (Need - Risk)')
    fig.suptitle(title + " " + metric + ".")


def main():
    """Main function to run simulations and generate plots."""
    c = Client(ENGINE_PATH=str(ENGINE_PATH))
    c.start_server()

    #This search is for the entire param range, so the same for all explorations
    search = {'W': {'start': 1, 'stop': 50, 'num': 50},
              'fixed_lambda': {'start': 0, 'stop': 10, 'num': 50},
              'fixed_rho': {'start': 0, 'stop': 10, 'num': 50},
              'fixed_eta': {'start': 0, 'stop': 10, 'num': 50},
              'fixed_psi': {'start': 0, 'stop': 1, 'num': 50}}

    # To run new simulations, add parameter names to this list. e.g., ['W', 'fixed_lambda']
    params_to_scan = []
    explorations_to_scan = ['risk_ground']
    for exploration in explorations_to_scan:
        for param in params_to_scan:
            run_simulation_scan(client=c, base_params=get_Params(exploration=exploration), param_name=param,
                                start=search[param]['start'], stop=search[param]['stop'], num=search[param]['num'],
                                base_working_directory=BASE_WORKING_DIRECTORY,
                                search_name=exploration)

    # --- Analysis and Plotting ---
    explorations_to_analyze = ['middle_ground', 'need_ground', 'risk_ground']
    for exploration in explorations_to_analyze:
        print(f"\n--- Analyzing and plotting for exploration: {exploration} ---")
        Hneeds, Hrisks = {}, {}
        Tneeds, Trisks = {}, {}

        for parameter in search.keys():
            try:
                Hneeds[parameter], Tneeds[parameter] = load_simulation_data(
                    base_working_directory=BASE_WORKING_DIRECTORY, search_name=exploration, param_name=parameter,
                    policy='need', **search[parameter])
                Hrisks[parameter], Trisks[parameter] = load_simulation_data(
                    base_working_directory=BASE_WORKING_DIRECTORY, search_name=exploration, param_name=parameter,
                    policy='risk', **search[parameter])
            except FileNotFoundError as e:
                print(f"Could not load data for parameter '{parameter}': {e}")
                continue

        metrics_to_plot = {
            'Mean H': (Hneeds, Hrisks, lambda x: np.mean(x, axis=1)),
            'Variance H': (Hneeds, Hrisks, lambda x: np.var(x, axis=1)),
            'Mean T': (Tneeds, Trisks, lambda x: np.nanmean(x, axis=1)),
            'Variance T': (Tneeds, Trisks, lambda x: np.nanvar(x, axis=1)),
        }

        for metric_name, (data_need, data_risk, func) in metrics_to_plot.items():
            results = {
                param: {'need': func(data_need[param]), 'risk': func(data_risk[param])}
                for param in search.keys() if param in data_need
            }
            if not results:
                print(f"No results to plot for metric: {metric_name}")
                continue

            combinedBars(results=results, search=search,
                         title=f'Tweaking One Parameter at a Time from the {exploration} Exploration',
                         metric=metric_name, exploration=exploration)

    plt.show()


if __name__ == "__main__":
    main()
