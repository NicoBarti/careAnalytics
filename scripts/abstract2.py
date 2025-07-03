import numpy as np
from scripts.pairPlots import *
from MyClasses import plotting
from MyClasses.client import Client
import matplotlib.pyplot as plt
from scripts.noHetero.CAPACITY_DISEASE_specialPairPlots import attempts_needs_expectations

def do_abstract2():
    ENGINE_PATH="./engine/CareEngine5_socket_11020_manager.jar"
    c = Client(ENGINE_PATH=ENGINE_PATH)
    do_grid_manager(c)

    c = Client(ENGINE_PATH="./engine/CareEngine5_socket_noHetero.jar")
    do_grid_capacity_diseasec(c)

def do_grid_manager(c):
    c.start_server()

    lists = [{'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
              'values': np.arange(start=0, stop=40, step=2), 'gridParValue': 1, 'vline': '',
              'SUBJECTIVE_INITIATIVE': 0.3, 'SEVERITY_ALLOCATION':0,
              'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
              'anotate': '', 'expline': '', 'type': 'needs', 'ds': [1, 2, 3]},
             {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
              'values': np.arange(start=0, stop=40, step=2), 'gridParValue': 1, 'vline': '',
              'SUBJECTIVE_INITIATIVE': 0.3, 'SEVERITY_ALLOCATION': 1,
              'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
              'anotate': '', 'expline': '', 'type': 'needs', 'ds': [1, 2, 3]},
             ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='ghostwhite')
    i=0
    title = ['Base model', 'Outreach strategy (nudge)']
    for list in lists:
        c.start_server()
        parDic = ensamble_par({list['plotPar']: list['values']}, {list['gridPar']: [list['gridParValue']]})
        parDic['SUBJECTIVE_INITIATIVE'] = [list['SUBJECTIVE_INITIATIVE']]
        parDic['SEVERITY_ALLOCATION'] = [list['SEVERITY_ALLOCATION']]
        gridParameters = gridCombined(parDic)
        #data = c.socket_with_model_paramGrid_2(gridParameters)
        #metrics = Metrics_socket(data, gridParameters, False)
        N = 10
        errors = Errors_socket(gridParameters = gridParameters, N=N, client=c)
        p = Plotter()
        plot1 = p.one_error_plot2(dataObject=errors, axe=axes[i], plot_par_name= list['plotPar'],grid_par_name=list['gridPar'],
                    metric_name='need_evolution',sub_grid=gridParameters)
        plot1.get_lines()[0].set(linewidth=2)
        plot1.get_lines()[1].set(linewidth=2)
        plot1.get_lines()[2].set(linewidth=2)
        axes[i].set_title(title[i], size=16)
        axes[i].set_ylim(0,3.5)
        axes[i].set_xlabel('$\\alpha$', size=14)
        axes[i].legend(["$\\delta = 0.5$", "$\\delta = 2$", "$\\delta = 3$"])
        axes[i].set_xticks(np.arange(start=0, stop=40, step=5))
        axes[i].set_ylabel('Average needs at the end of the simulation', size=14)
        i=i+1
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.07)
    fig.show()
    fig.savefig('./figures/abstract/managerGrid.png')

def do_grid_capacity_diseasec(c):
    c.start_server()
    lists = [{'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
         'values': np.arange(start=0, stop=80, step=5), 'gridParValue': 2, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0.3,
         'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
         'anotate': '', 'expline': '', 'type': 'needs'},
             {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
              'values': np.arange(start=0, stop=18, step=1), 'gridParValue': 0.3, 'vline': '',
              'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
              'anotate': '', 'expline': '', 'type': 'needs'}
             ]
    parDic = ensamble_par({lists[0]['plotPar']: lists[0]['values']}, {lists[0]['gridPar']: [lists[0]['gridParValue']]})
    parDic['SUBJECTIVE_INITIATIVE'] = [lists[0]['SUBJECTIVE_INITIATIVE']]
    gridParameters = gridCombined(parDic)
    errors = Errors_socket(gridParameters=gridParameters, N=2, client=c)
    p = Plotter()
    fig, axes = plt.subplots(1,2, figsize=(16, 8), facecolor='ghostwhite')
    fig, axes[0] = attempts_needs_expectations(list=lists[0], c = c, p = p, errors=errors, gridParameters=gridParameters,
                                          axe = axes[0], fig = fig)
    axes[0].set_title(f"$\\delta = {lists[0]['gridParValue']}$", size=16)
    axes[0].set_xlabel('$\\alpha$', size=16)
    axes[0].set_ylim(0,30)
    fig.axes[2].yaxis.set_label_text('Average Needs / Expectations at the end of the simulation', color = "Black")
    fig.axes[2].tick_params(labelcolor = "black", color = "black")
    fig.axes[2].set_title("")
    fig.axes[3].yaxis.set_visible(False)
    fig.axes[3].set_title("")

    c.start_server()
    parDic = ensamble_par({lists[1]['plotPar']: lists[1]['values']}, {lists[1]['gridPar']: [lists[1]['gridParValue']]})
    parDic['LEARNING_RATE'] = [lists[1]['LEARNING_RATE']]
    parDic['capacity'] = [30]
    gridParameters = gridCombined(parDic)
    errors = Errors_socket(gridParameters=gridParameters, N=2, client=c, fetchType='add_progression')
    fig, axes[1] = attempts_needs_expectations(list=lists[1], c = c, p = p, errors=errors, gridParameters=gridParameters,
                                          axe = axes[1], fig = fig)
    axes[1].set_title(f"$\\alpha = {parDic['capacity'][0]}$", size=16)
    axes[1].set_xlabel('$\\delta$', size=16)
    axes[1].set_ylim(0,90)
    fig.axes[4].yaxis.set_label_text('Average Needs / Expectations at the end of the simulation', color = "Black")
    fig.axes[4].tick_params(labelcolor = "black", color = "black")
    fig.axes[4].set_title("")
    fig.axes[5].yaxis.set_visible(False)
    fig.axes[5].set_title("")
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.07)
    fig.savefig('./figures/abstract/disease_capacityGrid.png')
    fig.show()
