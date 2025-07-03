import numpy as np
from scipy.stats import alpha

from scripts.pairPlots import *
from MyClasses.client import *
from scripts.needs_to_attemps import *


def do_capacity_DISEASE_special_pairs(ENGINE_PATH, lists  = []):
    print('doing special_pairs')
    c = Client(ENGINE_PATH = ENGINE_PATH)
    p = Plotter()

    att_from_capacity_with_disease = [
         # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
         #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 5, 'vline': 60, 'SUBJECTIVE_INITIATIVE' : 0.5,
         #   'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
         #  'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 5, 'vline': 70, 'SUBJECTIVE_INITIATIVE': 0,
        #  'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 5, 'vline': 190, 'SUBJECTIVE_INITIATIVE': 1,
        #  'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'expectations'},
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 10, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0.5,
        #  'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 10, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0.2,
        #  'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 15, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0.5,
        #  'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        # {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
        #  'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 20, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0.5,
        #  'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
        #  'anotate': '', 'expline': '', 'type': 'needs'},
        {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
         'values': np.arange(start=0, stop=200, step=10), 'gridParValue': 10, 'vline': 80, 'SUBJECTIVE_INITIATIVE': 0.3,
         'attline': '', 'neeline': '', 'filename': 'abstract1', 'maxAtt': '', 'minAtt': '',
         'anotate': '', 'expline': '', 'type': 'needs'},
    ]

    if len(lists) == 0:
        lists = att_from_capacity_with_disease

    for list in lists:
        c.start_server()
        parDic = ensamble_par({list['plotPar']: list['values']}, {list['gridPar']: [list['gridParValue']]})
        parDic['SUBJECTIVE_INITIATIVE'] = [list['SUBJECTIVE_INITIATIVE']]
        gridParameters = gridCombined(parDic)
        errors = Errors_socket(gridParameters=gridParameters, N=10, client=c, fetchType='add_progression')
        if list['vline'] != '':
            detailFigure, Bmean, mean = do_mechanismNeedsToAttempts2(gridParameters=gridParameters, N=10, errors=errors, list=list,
                                                        model = c.get_model_name() ,
                                                        name = f'_D{list['gridParValue']}_S{list["SUBJECTIVE_INITIATIVE"]}_{list["filename"]}.png',
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
        filename = root + f'att_needexpect_D{list['gridParValue']}_S{list["SUBJECTIVE_INITIATIVE"]}_{list["filename"]}.png'
        fig.show()
        fig.savefig(filename)

        fig, axe = attempts_visits_treatment(list=list,c=c,p=p, errors=errors, gridParameters=gridParameters)
        #postProcess(fig=fig, axe=axe, list=list, errors=errors, gridParameters=gridParameters)
        filename = root + f'att_visits_D{list['gridParValue']}_S{list["SUBJECTIVE_INITIATIVE"]}_{list["filename"]}.png'
        fig.show()
        fig.savefig(filename)

        filename = root + f'needsPerStep__D{list['gridParValue']}_S{list["SUBJECTIVE_INITIATIVE"]}_{list["filename"]}.png'
        if list['vline'] != '':
            detailFigure.savefig(filename)

    return(None)

def do_capacity_DISEASE_triptics(ENGINE_PATH):
    c = Client(ENGINE_PATH = ENGINE_PATH)
    p = Plotter()

    lists = [    {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
     'values': np.append(np.arange(start=0, stop=50, step=5), np.arange(start = 50, stop = 251, step = 50)), 'gridParValue': 2, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0,
     'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
     'anotate': '', 'expline': '', 'type': 'needs'}, {'plotPar': 'capacity', 'gridPar': 'DISEASE_SEVERITY',
     'values': np.append(np.arange(start=0, stop=100, step=5), np.arange(start=100, stop=251, step=50)), 'gridParValue': 10, 'vline': '', 'SUBJECTIVE_INITIATIVE': 0,
     'attline': '', 'neeline': '', 'filename': 'zoomOut', 'maxAtt': '', 'minAtt': '',
     'anotate': '', 'expline': '', 'type': 'needs'}]
    figs = [None] * len(lists)
    fig, axe = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), facecolor='ghostwhite')
    for i in range(len(lists)):
        c.start_server()
        list = lists[i]
        parDic = ensamble_par({list['plotPar']: list['values']}, {list['gridPar']: [list['gridParValue']]})
        parDic['SUBJECTIVE_INITIATIVE'] = [list['SUBJECTIVE_INITIATIVE']]
        gridParameters = gridCombined(parDic)
        errors = Errors_socket(gridParameters=gridParameters, N=10, client=c, fetchType='add_progression')
        p.set_colors('black')
        #p.labels = ['Attempts','Attempts','Attempts']
        p.set_labels(False)
        p.one_error_plot2(dataObject=errors, axe=axe[i], plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                          metric_name='attempts_patient_simulation', sub_grid=gridParameters, bars=True,
                          colideGroups=False, ds=[1])
        axe[i].set(ylim=(0, 150))
        axe[i].set_xticks(np.arange(start=0, stop = 250, step = 25))
    fig.show()
    filename = './figures/' + c.get_model_name() + '/participation/' + list['plotPar'] + '__' + list['gridPar'] + '/' + 'att_diptic.png'
    fig.savefig(filename)

def attempts_visits_treatment(list, c, p, errors, gridParameters):
    """Plot attempts as the main variable, visits as the secondary variable and save the plot."""

    p.set_labels(False)

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), facecolor='ghostwhite')
    twin1 = axe.twinx()
    twin2 = axe.twinx()

    p.set_colors('black')
    l1 = mpl.lines.Line2D([], [], color=p.get_colors()[0], linestyle = p.get_linestyle()[0],label='Attempts')
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
    l2 = mpl.lines.Line2D([], [], color=p.get_colors()[0], linestyle = p.get_linestyle()[0], label='Visits')
    plot2 = p.one_error_plot2(dataObject=errors, axe=twin1, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='avg_visits', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot2.yaxis.label.set_color(p.get_colors()[0])
    maxVisits = int(np.round(gridParameters['capacity'].max() * gridParameters['weeks'].max()/ gridParameters['numPatients'].max()))
    plot2.set(ylim=(0, maxVisits))
    plot2.get_lines()[0].set(linewidth=2)

    plot2.tick_params(axis='y', colors=p.get_colors()[0])

    p.set_colors('progression')
    l3 = mpl.lines.Line2D([], [], color=p.get_colors()[0], linestyle = p.get_linestyle()[0], label='Needs generated')
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

def attempts_needs_expectations(list, c, p, errors, gridParameters, axe = None, fig = None):
    """Plot attempts as the main variable, needs and expectations as the secondary variable and save the plot."""
    p.set_labels(False)

    if axe == None:
        fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), facecolor='ghostwhite')
    twin1 = axe.twinx()
    twin2 = axe.twinx()
    if 'ds' in list:
        ds = list['ds']
    else:
        ds = [1]
    p.set_colors('black')
    l1 = mpl.lines.Line2D([], [], color=p.get_colors()[0], label='Attempts')
    plot1 = p.one_error_plot2(dataObject=errors, axe=axe, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='attempts_patient_simulation', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=ds)
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
    l2 = mpl.lines.Line2D([], [], color=p.get_colors()[0], label='Expectations', linestyle=p.get_linestyle()[0])
    plot2 = p.one_error_plot2(dataObject=errors, axe=twin1, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='expectation_endstage_nogroup', sub_grid=gridParameters, bars=True,
                              colideGroups=False, ds=[1])
    plot2.yaxis.label.set_color(p.get_colors()[0])
    plot2.set(ylim=(0, 5))
    plot2.tick_params(axis='y', colors=p.get_colors()[0])

    p.set_colors('needs')
    l3 = mpl.lines.Line2D([], [], color=p.get_colors()[0], label='Needs', linestyle=p.get_linestyle()[0])
    plot3 = p.one_error_plot2(dataObject=errors, axe=twin2, plot_par_name=list['plotPar'], grid_par_name=list['gridPar'],
                              metric_name='need_evolution_nogroup', sub_grid=gridParameters, bars=True, colideGroups=False,
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