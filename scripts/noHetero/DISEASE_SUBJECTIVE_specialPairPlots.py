import numpy as np
from scipy.stats import alpha

from scripts.pairPlots import *
from MyClasses.client import *
from scripts.needs_to_attemps import *

def do_disease_subjective_special_pairs(ENGINE_PATH):
    print('doing special_pairs')
    c = Client(ENGINE_PATH = ENGINE_PATH)
    p = Plotter()
    att_from_disease_with_subjective_1 = [
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=0, stop=40, step=5), 'gridParValue': 0, 'vline': 15,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=0, stop=40, step=5), 'gridParValue': 1, 'vline': 15,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type' :'expectations'},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=10, stop=13, step=0.5), 'gridParValue': 0, 'vline': 11,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'linearPart', 'maxAtt': 20, 'minAtt': 7,
        #  'anotate': None},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=11, stop=20, step=0.5), 'gridParValue': 0, 'vline': 15,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'accPart', 'maxAtt': 120, 'minAtt': 8,
        #  'anotate': None},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=30, stop=50, step=1), 'gridParValue': 0, 'vline': 35,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'satPart', 'maxAtt': '', 'minAtt': 20,
        #  'anotate': None},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=0, stop=40, step=5), 'gridParValue': 0.5, 'vline': '',
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
         'values': np.arange(start=0, stop=40, step=2), 'gridParValue': 0.3, 'vline': 10,
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
         'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=0, stop=40, step=2), 'gridParValue': 0.2, 'vline': '',
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
    ]
    att_from_disease_with_subjective_0 = [
        {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
         'values': np.sort(np.append(np.arange(start=0, stop=40, step=2), np.array([11, 15, 35]))), 'gridParValue': 0, 'vline': '',
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
         'anotate': [{'x': 11, 'text': 'A'}, {'x': 15, 'text': 'B'}, {'x': 35, 'text': 'C'}]},
        {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
         'values': np.arange(start=10, stop=13, step=0.5), 'gridParValue': 0, 'vline': 11,
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'linearPart', 'maxAtt': 20, 'minAtt': 7,
         'anotate': None},
        {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
         'values': np.arange(start=11, stop=20, step=0.5), 'gridParValue': 0, 'vline': 15,
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'accPart', 'maxAtt': 120, 'minAtt': 8,
         'anotate': None},
        {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
         'values': np.arange(start=30, stop=50, step=1), 'gridParValue': 0, 'vline': 35,
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'satPart', 'maxAtt': '', 'minAtt': 20,
         'anotate': None},
    ]

    att_from_capacity_with_disease = [
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=25), 'gridParValue': 4, 'vline': 25, 'SUBJECTIVE_INITIATIVE' : 0,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
         'values': np.arange(start=0, stop=40, step=5), 'gridParValue': 0, 'vline': 15,
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
         'anotate': '', 'expline': '', 'type' :'expectations'},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=0, stop=40, step=5), 'gridParValue': 1, 'vline': 15,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type' :'expectations'},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=10, stop=13, step=0.5), 'gridParValue': 0, 'vline': 11,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'linearPart', 'maxAtt': 20, 'minAtt': 7,
        #  'anotate': None},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=11, stop=20, step=0.5), 'gridParValue': 0, 'vline': 15,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'accPart', 'maxAtt': 120, 'minAtt': 8,
        #  'anotate': None},
        # {'plotPar': 'DISEASE_SEVERITY', 'gridPar': 'SUBJECTIVE_INITIATIVE',
        #  'values': np.arange(start=30, stop=50, step=1), 'gridParValue': 0, 'vline': 35,
        #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'satPart', 'maxAtt': '', 'minAtt': 20,
        #  'anotate': None},
    ]


    lists = att_from_disease_with_subjective_1


    for list in lists:
        c.start_server()
        parDic = ensamble_par({list['plotPar']: list['values']}, {list['gridPar']: [list['gridParValue']]})
        parDic['LEARNING_RATE'] = [list['LEARNING_RATE']]
        gridParameters = gridCombined(parDic)
        errors = Errors_socket(gridParameters=gridParameters, N=10, client=c, fetchType='add_progression')
        if list['vline'] != '':
            detailFigure, Bmean, mean = do_mechanismNeedsToAttempts2(gridParameters=gridParameters, N=10, errors=errors, list=list,
                                                        model = c.get_model_name() ,
                                                        name = f'_S{list['gridParValue']}_L{list["LEARNING_RATE"]}_{list["filename"]}.png',
                                                        type = list['type'])
            toAverageAttemptsPatients = Bmean*150/1000
            list['attline'] = toAverageAttemptsPatients
            toAveragePatients = mean / 1000
            if list['type'] == 'needs':
                list['neeline'] = toAveragePatients
            if list['type'] == 'expectations':
                list['expline'] = toAveragePatients

        root = './figures/' + c.get_model_name() + '/participation/' + list['plotPar'] + '__' + list['gridPar'] + '/'
        fig, axe = attempts_needs_expectations(list=list, c=c, p=p, errors=errors, gridParameters=gridParameters)
        filename = root + f'att_needexpect_S{list['gridParValue']}_L{list["LEARNING_RATE"]}_{list["filename"]}.png'
        fig.show()
        fig.savefig(filename)

        fig, axe = attempts_visits_treatment(list=list,c=c,p=p, errors=errors, gridParameters=gridParameters)
        #postProcess(fig=fig, axe=axe, list=list, errors=errors, gridParameters=gridParameters)
        filename = root + f'att_visits_S{list['gridParValue']}_L{list["LEARNING_RATE"]}_{list["filename"]}.png'
        fig.show()
        fig.savefig(filename)

        filename = root + f'needsPerStep__S{list['gridParValue']}_L{list["LEARNING_RATE"]}_{list["filename"]}.png'
        if list['vline'] != '':
            detailFigure.savefig(filename)

    return(None)

def attempts_visits_treatment(list, c, p, errors, gridParameters):
    """Plot attempts as the main variable, visits as the secondary variable and save the plot."""

    p.set_labels(False)

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), facecolor='ghostwhite')
    twin1 = axe.twinx()
    twin2 = axe.twinx()

    p.set_colors('black')
    l1 = mpl.lines.Line2D([], [], color=p.get_colors()[0], linestyle = p.get_linestyle(),label='Attempts')
    plot1 = p.one_error_plot2(dataObject=errors, axe=axe, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='attempts_patient_simulation', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot1.get_lines()[0].set(linewidth=2)
    if list['maxAtt'] == "":
        maxAttempts = int(np.round(gridParameters['weeks'][0]))
    else:
        maxAttempts = list['maxAtt']
    if list['minAtt'] == "":
        minAttempts = 0
    else:
        minAttempts = list['minAtt']
    plot1.set(ylim=(minAttempts, maxAttempts))
    plot1.yaxis.label.set_color(p.get_colors()[0])
    plot1.tick_params(axis='y', colors=p.get_colors()[0])

    p.set_colors('visits')
    l2 = mpl.lines.Line2D([], [], color=p.get_colors()[0], linestyle = p.get_linestyle(), label='Visits')
    plot2 = p.one_error_plot2(dataObject=errors, axe=twin1, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='avg_visits', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot2.yaxis.label.set_color(p.get_colors()[0])
    maxVisits = int(np.round(gridParameters['capacity'][0] * gridParameters['weeks'][0]/ gridParameters['numPatients'][0]))
    plot2.set(ylim=(0, maxVisits))
    plot2.get_lines()[0].set(linewidth=2)

    plot2.tick_params(axis='y', colors=p.get_colors()[0])

    p.set_colors('progression')
    l3 = mpl.lines.Line2D([], [], color=p.get_colors()[0], linestyle = p.get_linestyle(), label='Needs generated')
    plot3 = p.one_error_plot2(dataObject=errors, axe=twin2, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='mean_progress', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot3.yaxis.label.set_color(p.get_colors()[0])
    plot3.set(ylim=(minAttempts, maxAttempts))
    plot3.get_lines()[0].set(linewidth=2)

    plot3.tick_params(axis='y', colors=p.get_colors()[0])

    fig.subplots_adjust(right=0.75)
    twin2.spines.right.set_position(("axes", 1.2))
    axe.legend(handles=[l1, l2, l3], fontsize='large')


    if list['vline'] != "":
        axe.axvline(x=list['vline'], color='black', linestyle=':', lw = 0.8, alpha=0.8)
        newTicks = np.append(axe.get_xticks(), np.array(list['vline']))
        axe.set_xticks(newTicks)

    if list['attline'] != "":
        axe.axhline(y=list['attline'], color='black', linestyle=':', lw = 0.8, alpha=0.8)
        newTicks = np.append(axe.get_yticks(), np.array(list['attline']))
        axe.set_yticks(newTicks)

    plot1.set(ylim=(minAttempts, maxAttempts))
    return(fig, axe)

def attempts_needs_expectations(list, c, p, errors, gridParameters):
    """Plot attempts as the main variable, needs and expectations as the secondary variable and save the plot."""
    p.set_labels(False)

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), facecolor='ghostwhite')
    twin1 = axe.twinx()
    twin2 = axe.twinx()

    p.set_colors('black')
    l1 = mpl.lines.Line2D([], [], color=p.get_colors()[0], label='Attempts')
    plot1 = p.one_error_plot2(dataObject=errors, axe=axe, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='attempts_patient_simulation', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot1.get_lines()[0].set(linewidth=2)
    plot1.yaxis.label.set_color(p.get_colors()[0])
    plot1.tick_params(axis='y', colors=p.get_colors()[0])
    if list['maxAtt'] == "":
        maxAttempts = int(np.round(gridParameters['weeks'][0]))
    else:
        maxAttempts = list['maxAtt']
    if list['minAtt'] == "":
        minAttempts = 0
    else:
        minAttempts = list['minAtt']
    plot1.set(ylim=(minAttempts, maxAttempts))

    p.set_colors('expectations')
    l2 = mpl.lines.Line2D([], [], color=p.get_colors()[0], label='Expectations')
    plot2 = p.one_error_plot2(dataObject=errors, axe=twin1, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='mean_expectations_nogroup', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot2.yaxis.label.set_color(p.get_colors()[0])
    plot2.set(ylim=(0, 5))
    plot2.tick_params(axis='y', colors=p.get_colors()[0])

    p.set_colors('needs')
    l3 = mpl.lines.Line2D([], [], color=p.get_colors()[0], label='Needs')
    plot3 = p.one_error_plot2(dataObject=errors, axe=twin2, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='mean_needs_nogroup', sub_grid=gridParameters, bars=True, colideGroups=False,
                              ds=[1])
    plot3.yaxis.label.set_color(p.get_colors()[0])
    plot3.set(ylim=(0, 5))
    plot3.tick_params(axis='y', colors=p.get_colors()[0])

    fig.subplots_adjust(right=0.75)
    twin2.spines.right.set_position(("axes", 1.2))
    axe.legend(handles=[l1, l2, l3], fontsize='large')

    if list['vline'] != "":
        axe.axvline(x=list['vline'], color='black', linestyle=':', lw = 0.8, alpha=0.8)
        newTicks = np.append(axe.get_xticks(), np.array(list['vline']))
        axe.set_xticks(newTicks)
    if list['neeline'] != "":
        plot3.axhline(y=list['neeline'], color='black', linestyle=':', lw = 0.8, alpha=0.8)
        newTicks = np.append(plot3.get_yticks(), np.array(list['neeline']))
        plot3.set_yticks(newTicks)
    if list['expline'] != "":
        plot2.axhline(y=list['expline'], color='black', linestyle=':', lw = 0.8, alpha=0.8)
        newTicks = np.append(plot2.get_yticks(), np.array(list['expline']))
        plot2.set_yticks(newTicks)
    return(fig, axe)

def postProcess(fig, axe, list, errors, gridParameters):
    """Post-process the plot."""
    if list['anotate'] != None:
        for i in range(len(list['anotate'])):
            try:
                nrow = gridParameters.loc[gridParameters[list['plotPar']] == list['anotate'][i]['x']].index[0]
            except IndexError as e:
                print('No such points found in the grid to annotate!')
                print(e)
            axe.annotate(list['anotate'][i]['text'], xy=(list['anotate'][i]['x'],errors.getAverageAtt(nrow=nrow)[:,0].mean()), xytext=(list['anotate'][i]['x'], errors.getAverageAtt(nrow=nrow)[:,0].mean()+5))
            axe.plot(x = list['anotate'][i]['x'], y = errors.getAverageAtt(nrow=nrow)[:,0].mean(), marker = 'X', color = 'black', markersize = 10)

    return(fig, axe)