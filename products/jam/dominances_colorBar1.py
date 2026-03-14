from MyClasses.pathFinder import PathFinder
from MyClasses.client import Client
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
base_working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dominance/'

## Will build the box-params + hist profiles for every .stat on the sub_path directory
base_java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/'
#sub_path = 'agents/bi_objective/'
#sub_path = 'design/norm_kurtExp_inequality/'
sub_path = 'JAMPaper/'


# A base param grid with fixed values
def get_Params(exploration:str):
    if exploration == "middle_ground":
        return {
        #fixed:
        'varsigma': [100],
        'OBS_PERIOD': [100], #only interested in the end value
        'N': [3400],
        'totalCapacity': [170],
        'random_delta_min': [0],
        'random_delta_max': [10],
        'fixed_tau': [2],
        'fixed_capN': [10],
        'fixed_capE': [10],
        'fixed_kappa': [0.5],
        #to be varied:
        'W': [25],
        'fixed_lambda': [4],
        'fixed_rho': [7],
        'fixed_eta': [8],
        'fixed_psi': [0.5],
        'Pi': ['basal'],
        'reproduce_line': ['true'],
        'obsH': ['true'],
        'obsT': ['true']
    }
    elif exploration == "need_ground":
        return {
            # fixed:
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
            # to be varied:
            'W': [35],
            'fixed_lambda': [6],
            'fixed_rho': [9],
            'fixed_eta': [9],
            'fixed_psi': [0.9],
            'Pi': ['basal'],
            'reproduce_line': ['true'],
            'obsH': ['true'],
            'obsT': ['true']
        }
    elif exploration == "risk_ground":
        return{
            # fixed:
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
            # to be varied:
            'W': [11],
            'fixed_lambda': [1],
            'fixed_rho': [5],
            'fixed_eta': [7],
            'fixed_psi': [0.1],
            'Pi': ['basal'],
            'reproduce_line': ['true'],
            'obsH': ['true'],
            'obsT': ['true']
        }

def run_simulation_scan(client, base_params, param_name, start, stop, num, base_working_directory, search_name):
    save_path = f'{base_working_directory}/{search_name}/{param_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
                print("error inesperado!")

            Hs= np.concatenate([Hs, [np.array(run_data[0]['H'])[:,1]]], axis=0)
            Ts =np.concatenate([Ts, [np.array(run_data[0]['T'])[:, 1]]], axis=0)

        np.save(file= f'{save_path}/{policy}_Hs_start{start}_stop{stop}_num{num}', arr = Hs)
        np.save(file= f'{save_path}/{policy}_Ts_start{start}_stop{stop}_num{num}', arr = Ts)

        #Hs = pd.DataFrame(Hs, columns = [x for x in range(3400)], index = param_values)
        #Ts = pd.DataFrame(Ts, columns = [x for x in range(3400)], index = param_values)

        #Hs.to_csv(f'{save_path}/{policy}_Hs_start{start}_stop{stop}_num{num}', index=False)
        #Ts.to_csv(f'{save_path}/{policy}_Ts_start{start}_stop{stop}_num{num}', index=False)
    
def load_simulation_data(base_working_directory, search_name, param_name, policy, start, stop, num):
    """
    Loads the .npy files for a given simulation configuration.
    
    Returns:
        tuple: (Hs, Ts) arrays loaded from the files.
    """
    load_path = f'{base_working_directory}/{search_name}/{param_name}'
    hs_file = f'{load_path}/{policy}_Hs_start{start}_stop{stop}_num{num}.npy'
    ts_file = f'{load_path}/{policy}_Ts_start{start}_stop{stop}_num{num}.npy'
    
    if not os.path.exists(hs_file) or not os.path.exists(ts_file):
        raise FileNotFoundError(f"Data files not found in {load_path}")
        
    Hs = np.load(hs_file)
    Ts = np.load(ts_file)
    Ts[Ts == 0] = np.nan
    return Hs, Ts

def combinedBars(results, search,title, metric, exploration):
    #global color norm:
    dominances = {}
    for param in results.keys():
        dominances[param] = results[param]['need'] - results[param]['risk']
    norm = mcolors.CenteredNorm(halfrange=np.nanmax(np.abs(np.array(list(dominances.values())))))
    cmap = plt.get_cmap('PRGn')

    #reference base parameters
    ref_params = get_Params(exploration)

    #a horizontal grid of param columns
    fig, axe = plt.subplots(figsize=(9.2, 5), ncols=1, nrows=len(results.keys()), layout="constrained")
    for ax, par in zip(axe, results.keys()):
        ax.set_xlim(left=search[par]['start'], right= search[par]['stop'])
        widths = (search[par]['stop'] - search[par]['start']) / search[par]['num']
        ax.set_xlabel(par)
        for segment, dom in zip(np.linspace(start=search[par]['start'], stop=search[par]['stop'],
                                            num=search[par]['num']), dominances[par]):
            starts = segment
            rects = ax.barh(par, widths, left=starts, height=0.5, color=cmap(norm(dom)))
        #add reference param valua (base param)
        ax.vlines(ref_params[par], 0, 1)



    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axe)
    cbar.set_label(f'Difference in {metric} between Strategies (Need - Risk)')
    fig.suptitle(title + " "+ metric+".")
    fig.show()

c = Client(ENGINE_PATH=ENGINE_PATH)
c.start_server()

search = {'W' : {'start':1, 'stop':50, 'num':50},
          'fixed_lambda' : {'start':0, 'stop':10, 'num':50},
          'fixed_rho' : {'start':0, 'stop':10, 'num':50},
          'fixed_eta' : {'start':0, 'stop':10, 'num':50},
          'fixed_psi' : {'start':0, 'stop':1, 'num':50}}

params = []

exploration = 'middle_ground'
for param in params:
    run_simulation_scan(client=c, base_params=get_Params(exploration=exploration), param_name=param,
                    start=search[param]['start'], stop=search[param]['stop'], num=search[param]['num'],
                    base_working_directory=base_working_directory,
                    search_name=exploration)

Hneeds, Hrisks = {},{}
Tneeds, Trisks = {},{}

for parameter in search.keys():
    Hneeds[parameter],Tneeds[parameter] = load_simulation_data(base_working_directory=base_working_directory, search_name= exploration, param_name=parameter,
                          policy='need', start=search[parameter]['start'], stop=search[parameter]['stop'], num=search[parameter]['num'])
    Hrisks[parameter],Trisks[parameter] = load_simulation_data(base_working_directory=base_working_directory, search_name= exploration, param_name=parameter,
                          policy='risk', start=search[parameter]['start'], stop=search[parameter]['stop'], num=search[parameter]['num'])


results = {
    'W': {'need': Hneeds['W'].mean(1), 'risk':Hrisks['W'].mean(1)},
    'fixed_lambda': {'need': Hneeds['fixed_lambda'].mean(1), 'risk': Hrisks['fixed_lambda'].mean(1)},
    'fixed_rho' : {'need': Hneeds['fixed_rho'].mean(1), 'risk': Hrisks['fixed_rho'].mean(1)},
    'fixed_eta' : {'need': Hneeds['fixed_eta'].mean(1), 'risk': Hrisks['fixed_eta'].mean(1)},
    'fixed_psi' : {'need': Hneeds['fixed_psi'].mean(1), 'risk': Hrisks['fixed_psi'].mean(1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.'
             , metric ='Mean H', exploration=exploration)

results = {
    'W': {'need': Hneeds['W'].var(1), 'risk':Hrisks['W'].var(1)},
    'fixed_lambda': {'need': Hneeds['fixed_lambda'].var(1), 'risk': Hrisks['fixed_lambda'].var(1)},
    'fixed_rho' : {'need': Hneeds['fixed_rho'].var(1), 'risk': Hrisks['fixed_rho'].var(1)},
    'fixed_eta' : {'need': Hneeds['fixed_eta'].var(1), 'risk': Hrisks['fixed_eta'].var(1)},
    'fixed_psi' : {'need': Hneeds['fixed_psi'].var(1), 'risk': Hrisks['fixed_psi'].var(1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Variance H', exploration = exploration)

results = {
    'W': {'need': np.nanmean(Tneeds['W'], axis=1) , 'risk':np.nanmean(Trisks['W'], axis=1)},
    'fixed_lambda': {'need': np.nanmean(Tneeds['fixed_lambda'], axis=1), 'risk': np.nanmean(Trisks['fixed_lambda'], axis=1)},
    'fixed_rho' : {'need': np.nanmean(Tneeds['fixed_rho'], axis=1), 'risk': np.nanmean(Trisks['fixed_rho'], axis=1)},
    'fixed_eta' : {'need': np.nanmean(Tneeds['fixed_eta'], axis=1), 'risk': np.nanmean(Trisks['fixed_eta'], axis=1)},
    'fixed_psi' : {'need': np.nanmean(Tneeds['fixed_psi'], axis=1), 'risk': np.nanmean(Trisks['fixed_psi'], axis=1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Mean T', exploration=exploration)

results = {
    'W': {'need': np.nanvar(Tneeds['W'], axis=1), 'risk':np.nanvar(Trisks['W'], axis=1)},
    'fixed_lambda': {'need': np.nanvar(Tneeds['fixed_lambda'], axis=1), 'risk': np.nanvar(Trisks['fixed_lambda'], axis=1)},
    'fixed_rho' : {'need': np.nanvar(Tneeds['fixed_rho'], axis=1), 'risk': np.nanvar(Trisks['fixed_rho'], axis=1)},
    'fixed_eta' : {'need': np.nanvar(Tneeds['fixed_eta'], axis=1), 'risk': np.nanvar(Trisks['fixed_eta'], axis=1)},
    'fixed_psi' : {'need': np.nanvar(Tneeds['fixed_psi'], axis=1), 'risk': np.nanvar(Trisks['fixed_psi'], axis=1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Variance T', exploration=exploration)

params = []

exploration = 'need_ground'
for param in params:
    run_simulation_scan(client=c, base_params=get_Params(exploration=exploration), param_name=param,
                    start=search[param]['start'], stop=search[param]['stop'], num=search[param]['num'],
                    base_working_directory=base_working_directory,
                    search_name=exploration)

Hneeds, Hrisks = {},{}
Tneeds, Trisks = {},{}

for parameter in search.keys():
    Hneeds[parameter],Tneeds[parameter] = load_simulation_data(base_working_directory=base_working_directory, search_name= exploration, param_name=parameter,
                          policy='need', start=search[parameter]['start'], stop=search[parameter]['stop'], num=search[parameter]['num'])
    Hrisks[parameter],Trisks[parameter] = load_simulation_data(base_working_directory=base_working_directory, search_name= exploration, param_name=parameter,
                          policy='risk', start=search[parameter]['start'], stop=search[parameter]['stop'], num=search[parameter]['num'])

results = {
    'W': {'need': Hneeds['W'].mean(1), 'risk':Hrisks['W'].mean(1)},
    'fixed_lambda': {'need': Hneeds['fixed_lambda'].mean(1), 'risk': Hrisks['fixed_lambda'].mean(1)},
    'fixed_rho' : {'need': Hneeds['fixed_rho'].mean(1), 'risk': Hrisks['fixed_rho'].mean(1)},
    'fixed_eta' : {'need': Hneeds['fixed_eta'].mean(1), 'risk': Hrisks['fixed_eta'].mean(1)},
    'fixed_psi' : {'need': Hneeds['fixed_psi'].mean(1), 'risk': Hrisks['fixed_psi'].mean(1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Mean H', exploration=exploration)

results = {
    'W': {'need': Hneeds['W'].var(1), 'risk':Hrisks['W'].var(1)},
    'fixed_lambda': {'need': Hneeds['fixed_lambda'].var(1), 'risk': Hrisks['fixed_lambda'].var(1)},
    'fixed_rho' : {'need': Hneeds['fixed_rho'].var(1), 'risk': Hrisks['fixed_rho'].var(1)},
    'fixed_eta' : {'need': Hneeds['fixed_eta'].var(1), 'risk': Hrisks['fixed_eta'].var(1)},
    'fixed_psi' : {'need': Hneeds['fixed_psi'].var(1), 'risk': Hrisks['fixed_psi'].var(1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Variance H', exploration=exploration)

results = {
    'W': {'need': np.nanmean(Tneeds['W'], axis=1) , 'risk':np.nanmean(Trisks['W'], axis=1)},
    'fixed_lambda': {'need': np.nanmean(Tneeds['fixed_lambda'], axis=1), 'risk': np.nanmean(Trisks['fixed_lambda'], axis=1)},
    'fixed_rho' : {'need': np.nanmean(Tneeds['fixed_rho'], axis=1), 'risk': np.nanmean(Trisks['fixed_rho'], axis=1)},
    'fixed_eta' : {'need': np.nanmean(Tneeds['fixed_eta'], axis=1), 'risk': np.nanmean(Trisks['fixed_eta'], axis=1)},
    'fixed_psi' : {'need': np.nanmean(Tneeds['fixed_psi'], axis=1), 'risk': np.nanmean(Trisks['fixed_psi'], axis=1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Mean T', exploration=exploration)

results = {
    'W': {'need': np.nanvar(Tneeds['W'], axis=1), 'risk':np.nanvar(Trisks['W'], axis=1)},
    'fixed_lambda': {'need': np.nanvar(Tneeds['fixed_lambda'], axis=1), 'risk': np.nanvar(Trisks['fixed_lambda'], axis=1)},
    'fixed_rho' : {'need': np.nanvar(Tneeds['fixed_rho'], axis=1), 'risk': np.nanvar(Trisks['fixed_rho'], axis=1)},
    'fixed_eta' : {'need': np.nanvar(Tneeds['fixed_eta'], axis=1), 'risk': np.nanvar(Trisks['fixed_eta'], axis=1)},
    'fixed_psi' : {'need': np.nanvar(Tneeds['fixed_psi'], axis=1), 'risk': np.nanvar(Trisks['fixed_psi'], axis=1)}
}
combinedBars(results = results, search=search, title=f'Tewaking One Parameter at a Time from the {exploration} Exploration.',
             metric ='Variance T', exploration=exploration)

