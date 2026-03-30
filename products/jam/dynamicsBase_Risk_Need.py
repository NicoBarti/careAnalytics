from MyClasses.linePlotter import LinePlotter
from MyClasses.trajectoryDistances import TrajectoryDistances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the reusable plotting functions from the new module
from products.outcomesMatrix.complex_plotter import complexAxeDict, populate_axe

p = LinePlotter()

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder7.jar.jar'
java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
# working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design/'
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/dynamicBase_Risk_Need/'

corr_inxex={}
prop_index={}
for ground in [ 'needGround2', 'riskGround2']:
  for pi in ['basal', 'risk', 'need']:
    varsigma = 200
    data = pd.read_csv(f'{working_directory}/{ground}/{ground}_{pi}.csv')
    distances = TrajectoryDistances(working_directory=f'{working_directory}/{ground}', allRuns_csv=data,
                                    ENGINE_PATH=ENGINE_PATH,
                                    varsigma=varsigma, selectionName=pi, minH=0, maxH=1000,
                                    OBS_PERIOD=1, oederByWindow=10,
                                    orderByVariable='H', norms={"totalCapacity": 170, "fixed_tau": 2, "Pi": 3},
                                    addParam={'Pi'})
    stateVariables= ['H', 'N', 'E', 'SimpleB', 'SimpleC','T', 'Disease', 'ExpNoise', 'InstExp']
    seed = 62920828945
    dd = distances.produce(stateVariables=stateVariables, selectionName=pi, seeds=[seed])[seed]
    params = p.printableParams(distances.get_seedParams(seed=seed, selectionName=pi, filterParams=False),
                               paramsPerLine=8, policy=True)

    allData = np.quantile(np.array(dd['H'][str(varsigma)]), q=[1])[0]
    ax_dict = complexAxeDict()
    populate_axe(ax_dict, dd=dd, low_h_cut=0, high_h_cut=allData, sub_title=params)

    print(f"Final mean H in the whole population {pi}: {dd['H'][str(varsigma)].mean()}")
    print(f"Final var H in the whole population {pi}: {dd['H'][str(varsigma)].var()}")

    print(f"Final mean E in the whole population {pi}: {dd['E'][str(varsigma)].mean()}")
    print(f"Final var E in the whole population {pi}: {dd['E'][str(varsigma)].var()}")
    
    # Get the figure from any of the axes in the mosaic to apply suptitle
    fig = next(iter(ax_dict.values())).get_figure()
    fig.suptitle(f'{ground} seed: {seed} {pi}', fontsize=50)
    plt.show()

    cor = []
    prop = []
    for i in range(201):
        cor.append(np.corrcoef(dd['SimpleC'].to_numpy()[:, i], dd['H'].to_numpy()[:, i])[0, 1])
        #An index from 0 to 170. 170 means all the more severe patients were prioritized on this step
        higher170 = np.argpartition(-dd['H'].to_numpy()[:, i], kth=170)[:170]
        prop.append(np.take(a=dd['SimpleC'].to_numpy()[:, i], indices=higher170).sum()/170)  #the proportion of the 170 more severe that received care
    corr_inxex[pi] = cor
    prop_index[pi] = prop

  fig2, axe2 = plt.subplots(1, 1)
  fig3, axe3 = plt.subplots(1,1)
  for key in corr_inxex.keys():
    axe2.plot(corr_inxex[key], label=f'{key}')
    axe3.plot(prop_index[key], label=f'{key}')
  axe2.legend()
  axe3.legend()
  fig2.suptitle(f'Correlations for {ground}')
  fig2.show()
  fig3.suptitle(f'Proportions for {ground}')
  fig3.show()






