#from zipimport import alt_path_sep
from sys import base_exec_prefix

from MyClasses.trajectory import Trajectory
from MyClasses.linePlotter import LinePlotter
from MyClasses.pathFinderCollector import PathFinderCollector
from MyClasses.trajectoryDistances import TrajectoryDistances
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
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
BASE_WORKING_DIRECTORY = HOME_DIR / 'Desktop/simulationOutputs/JAMPaper/deciles/'

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

    for policy in ['risk', 'need', 'basal']:
        Hs, Ts, Cs = np.empty((0, base_params['N'][0])), np.empty((0, base_params['N'][0])),np.empty((0, base_params['N'][0]))
        for val in param_values:
            base_params[param_name] = [val]
            base_params['Pi'] = [policy]
            base_params['W'] = [int(base_params['W'][0])]
            base_params['OBS_PERIOD'] = [int(base_params['OBS_PERIOD'][0])]
            base_params['obsSimpleC'] = ['true']

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
            Cs = np.concatenate([Ts, [np.array(run_data[0]['SimpleC'])[:, 1]]], axis=0)

        file_name_base = f'{policy}_start{start}_stop{stop}_num{num}'
        np.save(save_path / f'{file_name_base}_Hs.npy', arr=Hs)
        np.save(save_path / f'{file_name_base}_Ts.npy', arr=Ts)
        np.save(save_path / f'{file_name_base}_Cs.npy', arr=Cs)


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
    cs_file = load_path / f'{file_name_base}_Cs.npy'


    if not hs_file.exists() or not ts_file.exists() or not cs_file.exists():
        raise FileNotFoundError(f"Data files not found in {load_path}")

    Hs = np.load(hs_file)
    Ts = np.load(ts_file)
    Ts[Ts == 0] = np.nan
    Cs = np.load(cs_file)
    return Hs, Ts, Cs

def decilesPlots(data: list, decile: float, reference_window: int, title: str, xlab, type = 'lower', xmax=None):
# Use last step for reference deciles
    decile_cut = np.quantile(data[reference_window, :], q = decile)
    decile_mask = data[reference_window, :] <= decile_cut if type == 'lower' else data[reference_window, :] > decile_cut
    fig, axe = plt.subplots(ncols=data.shape[0], nrows=1, figsize=(30, 8), layout = 'constrained')
    for ax, Hs in zip(axe, data):
        #Use same hist for both lower decile and other deciles:
        hist, bin_edges = np.histogram(Hs)
        low = np.histogram(Hs[decile_mask], bins=bin_edges)[0]
        high = np.histogram(Hs[~decile_mask], bins=bin_edges)[0]
        ax.bar(bin_edges[:-1], low, color = 'blue', alpha = 0.5, label = 'Lower decile')
        ax.bar(bin_edges[:-1], high, color = 'green', alpha = 0.5, label = 'Higher 9 deciles')
        # ax.hist(Hs[decile_mask], color = 'blue', alpha = 0.5, label = 'Lower decile', density = True)
        # ax.hist(Hs[~decile_mask], color = 'green', alpha = 0.5, label = 'Higher 9 deciles', density = True)
        ax.set_ylabel('Number of patients', fontsize=15)
        ax.set_xlabel(xlab, fontsize=15)
        ax.set_xlim(0,xmax)
        ax.legend(fontsize=15)
    fig.suptitle(title, fontsize=30)
    fig.show()

c = Client(ENGINE_PATH=str(ENGINE_PATH))
c.start_server()


# run_simulation_scan(client=c, base_params=get_Params(exploration='need_ground'), param_name='OBS_PERIOD',
#                     start =1, stop = 100, num = 100, base_working_directory= BASE_WORKING_DIRECTORY,
#                     search_name='need_ground')

#run_simulation_scan(client=c, base_params=get_Params(exploration='risk_ground'), param_name='OBS_PERIOD',
#                     start =1, stop = 100, num = 5, base_working_directory= BASE_WORKING_DIRECTORY,
#                     search_name='risk_ground')

need_ground_Hs_need, need_ground_Ts_need, need_ground_Cs_need = load_simulation_data(base_working_directory=BASE_WORKING_DIRECTORY, search_name="need_ground",
                                    param_name="OBS_PERIOD", policy="need", start=1, stop=100, num=100)

need_ground_Hs_risk, need_ground_Ts_risk, need_ground_Cs_risk = load_simulation_data(base_working_directory=BASE_WORKING_DIRECTORY, search_name="need_ground",
                                    param_name="OBS_PERIOD", policy="risk", start=1, stop=100, num=100)

need_ground_Hs_basal, need_ground_Ts_basal, need_ground_Cs_basal = load_simulation_data(base_working_directory=BASE_WORKING_DIRECTORY, search_name="need_ground",
                                    param_name="OBS_PERIOD", policy="basal", start=1, stop=100, num=100)

# risk_ground_Hs_risk, risk_ground_Ts_risk, risk_ground_Cs_risk = load_simulation_data(base_working_directory=BASE_WORKING_DIRECTORY, search_name="risk_ground",
#                                     param_name="OBS_PERIOD", policy="risk", start=1, stop=100, num=5)
# risk_ground_Hs_need, risk_ground_Ts_need, risk_ground_Cs_need = load_simulation_data(base_working_directory=BASE_WORKING_DIRECTORY, search_name="risk_ground",
#                                     param_name="OBS_PERIOD", policy="need", start=1, stop=100, num=5)




decile = 0.1
refwin=0
# for data, title in zip([need_ground_Hs_need, need_ground_Hs_risk, need_ground_Hs_basal], ['Need Ground - Need', 'Need Ground - Risk', 'Need Ground - Basal']):
#     decilesPlots(data = np.take(data, [20,40,60,99], axis =0), decile=decile, reference_window = refwin, title = title)
# for data, title in zip([need_ground_Hs_need, need_ground_Hs_risk, need_ground_Hs_basal], ['Need Ground - Need', 'Need Ground - Risk', 'Need Ground - Basal']):
#     decilesPlots(data = np.take(data, [20,40,60,99], axis =0), decile=decile, reference_window = refwin, title = title, type="upper")

datas = {'Hs': [need_ground_Hs_need, need_ground_Hs_risk, need_ground_Hs_basal],
         'Cs':[np.cumsum(need_ground_Cs_need, axis=0), np.cumsum(need_ground_Cs_risk, axis=0), np.cumsum(need_ground_Cs_basal, axis=0)]}

for data, title, xlab in zip(datas['Cs'], ['Need Ground - Need', 'Need Ground - Risk', 'Need Ground - Basal'], ['Appointments']*3):
    decilesPlots(data = np.take(data, [20,40,60,99], axis =0), decile=decile, reference_window = refwin, title = title,
                 xlab=xlab, xmax=np.array(datas['Cs']).max())

for data, title, xlab in zip(datas['Hs'], ['Need Ground - Need', 'Need Ground - Risk', 'Need Ground - Basal'], ['H']*3):
    decilesPlots(data = np.take(data, [20,40,60,99], axis =0), decile=decile, reference_window = refwin, title = title,
                 xlab=xlab,xmax=np.array(datas['Hs']).max())



# decilesPlots(data = np.take(need_ground_Hs_need, [20,40,60,99], axis =0), decile=decile, reference_window = refwin, title = 'Need Ground - Need')
# decilesPlots(data = np.take(need_ground_Hs_risk, [20,40,60,99], axis=0), decile=decile, reference_window = refwin, title = 'Need Ground - Risk')
# decilesPlots(data = np.take(need_ground_Hs_basal, [20,40,60,99],axis=0), decile=decile, reference_window = refwin, title = 'Need Ground - Basal')



# decilesPlots(data = risk_ground_Hs_need[1:5,:], decile=decile, reference_window = refwin, title = 'Risk Ground - Need')
# decilesPlots(data = risk_ground_Hs_risk[1:5,:], decile=decile, reference_window = refwin, title = 'Risk Ground -Risk')
# decilesPlots(data = need_ground_Ts_need[1:5,:], decile=decile, reference_window = refwin, title = 'Need Ground - Need')
# decilesPlots(data = need_ground_Ts_risk[1:5,:], decile=decile, reference_window = refwin, title = 'Need Ground - Risk')
# decilesPlots(data = risk_ground_Ts_need[1:5,:], decile=decile, reference_window = refwin, title = 'Risk Ground - Need')
# decilesPlots(data = risk_ground_Ts_risk[1:5,:], decile=decile, reference_window = refwin, title = 'Risk Ground -Risk')
# decilesPlots(data = need_ground_Cs_need[1:5,:], decile=decile, reference_window = refwin, title = 'Need Ground - Need')
# decilesPlots(data = need_ground_Cs_risk[1:5,:], decile=decile, reference_window = refwin, title = 'Need Ground - Risk')
# decilesPlots(data = risk_ground_Cs_need[1:5,:], decile=decile, reference_window = refwin, title = 'Risk Ground - Need')
# decilesPlots(data = risk_ground_Cs_risk[1:5,:], decile=decile, reference_window = refwin, title = 'Risk Ground -Risk')