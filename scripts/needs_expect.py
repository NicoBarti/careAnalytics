from scripts.pairPlots import *
from MyClasses.client import *
from MyClasses.plotting import *

def plot_needs_expect(p, metrics, plot_par_name, grid_par_name, grid_par_value, filename, gridParameters):
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), facecolor='ghostwhite')

    p.set_colors([0.15,0.15,0.15])
    axe = p.one_plot2(dataObject = metrics, axe = axe, plot_par_name = plot_par_name, grid_par_name = grid_par_name, metric_name = 'mean_needs_nogroup', sub_grid = gridParameters, ds=[1],
                    variability=True)

    print('mean_expectations')
    p.set_colors([0.85,0.85,0.85])
    axe = p.one_plot2(dataObject = metrics, axe = axe, plot_par_name = plot_par_name, grid_par_name = grid_par_name, metric_name = 'mean_expectations_nogroup', sub_grid = gridParameters, ds=[1],
                    variability=True)

    axe.legend(axe.get_lines(), ["Needs","Expectations"], fontsize = 'large')
    axe.set_ylabel('Average value of the state variable')
    axe.set_ylim(-0.2,5.2)
    fig.show()
    fig.savefig(filename)
    return None

def do_needs_expect_plots(ENGINE_PATH = './engine/CareEngine5_socket.jar'):
    c = Client(ENGINE_PATH=ENGINE_PATH)
    plotter = Plotter(cmap = mpl.colormaps['PiYG'])
    print('doing needs_expect_plots')
    lists = [
             #['capacity','SUBJECTIVE_INITIATIVE',np.arange(start=0, stop=1400, step=80),1],
             ['capacity', 'SUBJECTIVE_INITIATIVE', np.arange(start=0, stop=1400, step=80), 0],
             #['capacity', 'DISEASE_VELOCITY', np.arange(start=0, stop=1400, step=80), 0],
             #['capacity', 'DISEASE_VELOCITY', np.arange(start=0, stop=1400, step=80), 9.6],
             #['DISEASE_VELOCITY', 'LEARNING_RATE', np.arange(start=0, stop=1400, step=80), 0],
             #['DISEASE_VELOCITY', 'LEARNING_RATE', np.arange(start=0, stop=1400, step=80), 1],
             #['DISEASE_VELOCITY', 'SUBJECTIVE_INITIATIVE', np.arange(start=0, stop=10, step=1.2), 0.03],
             #['DISEASE_VELOCITY', 'SUBJECTIVE_INITIATIVE', np.arange(start=0, stop=10, step=1.2), 1],
             #['LEARNING_RATE', 'SUBJECTIVE_INITIATIVE', np.arange(start=0, stop=1400, step=80), 0],
             #['LEARNING_RATE', 'SUBJECTIVE_INITIATIVE', np.arange(start=0, stop=1400, step=80), 1]
             ]
    for list in lists:
        filename = f'./figures/pairs/state_variables/needs_expe_{list[0]}_{list[1]}{list[3]}.png'
        gridParameters = configParams({list[0]: list[2]}, {list[1]: [list[3]]})
        c.start_server()
        data = c.socket_with_model_paramGrid_2(gridParameters)
        metrics = Metrics_socket(data, gridParameters, False)
        plot_needs_expect(metrics=metrics,plot_par_name=list[0], grid_par_name=list[1], grid_par_value=list[3],
                  filename=filename, gridParameters=gridParameters, p = plotter)
    return None

