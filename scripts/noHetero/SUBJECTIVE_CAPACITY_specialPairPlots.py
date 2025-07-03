from scripts.noHetero.DISEASE_SUBJECTIVE_specialPairPlots import *

def do_subjective_capacity_special_pairs(c):
    print('doing special_pairs')
    #c = Client(ENGINE_PATH = ENGINE_PATH)
    p = Plotter()

    att_from_subjective_with_capacity = [
         # {'plotPar': 'SUBJECTIVE_INITIATIVE', 'gridPar': 'capacity',
         #  'values': np.arange(start=0, stop=1.01, step=0.1), 'gridParValue': 0, 'vline': '', 'DISEASE_SEVERITY' : 3,
         #  'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'panoramic', 'maxAtt': '', 'minAtt': '',
         #  'anotate': '', 'expline': '', 'type': 'Distribute_needs', 'doVarPlot' : False},
        {'plotPar': 'SUBJECTIVE_INITIATIVE', 'gridPar': 'capacity',
         'values': np.arange(start=0, stop=1.01, step=0.1), 'gridParValue': 80, 'vline': 1, 'DISEASE_SEVERITY': 3,
         'LEARNING_RATE': 0, 'attline': '', 'neeline': '', 'filename': 'panoramic', 'maxAtt': '', 'minAtt': '',
         'anotate': '', 'expline': '', 'type': 'expectations', 'doVarPlot' : True, 'plotExpectationsHist': [0,0.5,1]},
    ]

    lists = att_from_subjective_with_capacity

    for list in lists:
        c.start_server()
        parDic = ensamble_par({list['plotPar']: list['values']}, {list['gridPar']: [list['gridParValue']]})
        parDic['LEARNING_RATE'] = [list['LEARNING_RATE']]
        parDic['DISEASE_SEVERITY'] = [list['DISEASE_SEVERITY']]
        gridParameters = gridCombined(parDic)
        errors = Errors_socket(gridParameters=gridParameters, N=10, client=c, fetchType='add_progression')
        if 'plotExpectationsHist' in list:
            plotExpectationsHist(errors=errors, model=c.get_model_name(), plot_values=list['plotExpectationsHist'])
        if list['doVarPlot']:
            do_varianceSubjective(errors=errors, model = c.get_model_name())
        if list['vline'] != '':
            name = f'_S{list['gridParValue']}_D{list["DISEASE_SEVERITY"]}_{list["filename"]}.png'
            if list['type'] == 'Distribute_needs':
                print('pending')
            detailFigure, Bmean, mean = do_mechanismNeedsToAttempts2(gridParameters=gridParameters, N=10, errors=errors, list=list,
                                                        model = c.get_model_name() ,
                                                        name = name,
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
        filename = root + f'att_needexpect_C{list['gridParValue']}_D{list["DISEASE_SEVERITY"]}_{list["filename"]}.png'
        fig.show()
        fig.savefig(filename)

        fig, axe = attempts_visits_treatment(list=list,c=c,p=p, errors=errors, gridParameters=gridParameters)
        #postProcess(fig=fig, axe=axe, list=list, errors=errors, gridParameters=gridParameters)
        filename = root + f'att_visits_C{list['gridParValue']}_D{list["DISEASE_SEVERITY"]}_{list["filename"]}.png'
        fig.show()
        fig.savefig(filename)

        filename = root + f'needsPerStep__C{list['gridParValue']}_D{list["DISEASE_SEVERITY"]}_{list["filename"]}.png'
        if list['vline'] != '':
            detailFigure.savefig(filename)
    return(None)


def do_varianceSubjective(errors, model):
    p = Plotter()
    gridParameters = errors.gridParameters
    data = errors.dic['metrics0']
    data.configureVariability('quantile')
    fig, axe = plt.subplots(1, 1, figsize=(8,8), facecolor='ghostwhite')
    p.one_plot2(dataObject = data, axe = axe, plot_par_name = "SUBJECTIVE_INITIATIVE", grid_par_name = "capacity",
                metric_name = 'attempts_patient_simulation', sub_grid = gridParameters, ds=[1],
                  variability=True)
    axe.legend().remove()
    root = './figures/' + model + '/participation/SUBJECTIVE_INITIATIVE__capacity/'
    fig.savefig(root + 'variability.png')
    fig.show()

def plotExpectationsHist(errors, model, plot_values):
    p = Plotter()
    fig, axes = plt.subplots(1, 3, figsize=(24,8), facecolor='ghostwhite')
    for i in range(0, len(plot_values)):
        metrics = errors.dic['metrics0']
        #fig, axe = plt.subplots(1, 1, figsize=(8, 8), facecolor='ghostwhite')
        p.one_needexpe_hist(dataObject=metrics, axe=axes[i], plot_par_name="SUBJECTIVE_INITIATIVE", grid_par_name="capacity",
                        sub_grid=errors.gridParameters, ds=[1], type="exp", plot_par_value=plot_values[i])
        root = './figures/' + model + '/participation/SUBJECTIVE_INITIATIVE__capacity/'
    fig.savefig(root+ f'expHist{plot_values}.png')
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.05)
    fig.show()
    return(None)



