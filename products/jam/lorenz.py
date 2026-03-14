import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MyClasses.trajectoryDistances import TrajectoryDistances

ENGINE_PATH = '/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/engine/PathFinder6_middleway.jar'
java_output = f'/Users/nicolasbarticevic/Desktop/ecj_run/q1_design/'
# working_directory = f'/Users/nicolasbarticevic/Desktop/ecj_run/simulationOutputs/otucomesMatrix/q1_design/'
working_directory = f'/Users/nicolasbarticevic/Desktop/simulationOutputs/readingGroup/'
varsigma = 100
treatment = 'hiPsi/basal'
data = pd.read_csv(f'/Users/nicolasbarticevic/Desktop/simulationOutputs/JAMPaper/{treatment}.csv')
selection = treatment
distances = TrajectoryDistances(working_directory=working_directory, allRuns_csv=data,
                                ENGINE_PATH=ENGINE_PATH,
                                varsigma=varsigma, selectionName=selection, minH=0, maxH=1000,
                                OBS_PERIOD=1, oederByWindow=10,
                                orderByVariable='H', norms={"totalCapacity": 200, "fixed_tau": 2, "Pi": 3},
                                addParam={'Pi'})

seed = 62920828945

parameters_recover = distances.get_seedParams(seed=seed, selectionName=selection, filterParams=False)
parameters_recover['seeds'] = parameters_recover['seed']


def plot_lorenz_curve(distances_obj, orderingState, seed, selection, ax, legend = 'Lorenz Curve',
                      title = 'Lorenz Curve', tweak = []):
    """
    Plot a Lorenz curve based on usage and ordered severity.

    Parameters:
    - distances_obj: TrajectoryDistances instance, properly initialized.
    - states: List of state variables to be processed.
    - seed: The seed to trace the required data.
    - selection: Treatment selection name.
    - ax: Optional matplotlib axis to draw on an existing figure.
    """
    # Produce results for given states and seed
    results = distances_obj.produce(stateVariables=['SimpleC', orderingState, 'H'], selectionName=selection, seeds=[seed],
                                    tweak=tweak)

    # Cumulative usage over time for 'SimpleC'
    usage = results[seed]['SimpleC'].cumsum(axis=1).to_numpy()
    totalUsage = usage.sum(0)[-1]  # Total usage at the end
    usagePerPatient = results[seed]['SimpleC'].cumsum(axis=1)['100']

    # Order severity
    def orderSeverity(value):
        if value == 'Delta':
            return results[seed]['Delta']['0'].sort_values()
        elif value == "H":
            return results[seed]['H'].iloc[:,-1].sort_values()
        else:
            return results[seed][value].sum(1).sort_values()

    orderedSeverity = orderSeverity(orderingState)

    # Lorenz curve values
    lorenz = usagePerPatient[orderedSeverity.index].cumsum() / totalUsage
    N = usagePerPatient.shape[0]
    equality = np.linspace(start=0, stop=1, num=N)

    def labelOrder(value):
        if value == 'Delta':
            return 'their individual disease progression rate (delta)'
        elif value == 'Disease':
            return 'the number of needs experienced during the simulation'
        elif value == 'N':
            return 'the sum of their needs trhoughout the simulation'
        elif value == 'H':
            return 'the health status at the end of the simulation'

    # Plotting
    def is_equity_line_present(ax):
        for line in ax.get_lines():
            if line.get_label() == 'Equality Line':
                return True
        return False

    ax.plot(range(N), equality, linestyle='--', label='Equality Line') if is_equity_line_present(ax) == False else None
    ax.plot(range(N), lorenz, label=legend)
    ax.set_title(title)
    ax.set_xlabel(f'Patients ordered by {labelOrder(orderingState)}')
    ax.set_ylabel('Cumulative percentaje of usage')
    ax.legend()
    meanH, varH =  results[seed]['H'].iloc[:,-1].mean(), results[seed]['H'].iloc[:,-1].var()
    return fig, ax,meanH, varH


capacities = np.linspace(10, 500, 10)
meanCapacity = []
varCapacity = []
fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Create new figure if none provided
for capacity in capacities:
    # Generate title dynamically for each plot
    legend = f'capacity = {int(capacity)}'
    title = f'Lorenz Curves for Increasing Capacity'
    # Call the function with the current psi value as a tweak parameter
    fig, ax, mean, var = plot_lorenz_curve(distances, orderingState='Disease', seed=seed, selection=selection, tweak={'totalCapacity': int(capacity)},
                      legend=legend, ax=ax, title = title)
    meanCapacity.append(mean)
    varCapacity.append(var)
fig.suptitle("Lorenz Curves for Increasing Capacity")
ax.set_title(f'Psi = 1')
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle("Mean and Variance of H for Different Capacity")
ax.set_title(f'Psi = 1')

line0 = ax.plot(capacities, meanCapacity, label='Mean H', color='red')
ax1 = ax.twinx()
line1 = ax1.plot(capacities, varCapacity, label='Variance H', color='blue')

# added these three lines
lns = line0+line1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)

ax.set_xlabel('Capacity')
ax.set_ylabel('Mean H')
ax1.set_ylabel('Var H')
fig.show()


psis = np.linspace(0, 1, 4)
meanPsis = []
varPsis = []
fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Create new figure if none provided
fixedCapacity = 64
for psi in psis:
    # Generate title dynamically for each plot
    legend = f'psi = {psi:.2f}'
    title = f'Lorenz Curves for Increasing Psi Values'
    # Call the function with the current psi value as a tweak parameter
    fig, ax, mean, var = plot_lorenz_curve(distances, orderingState='Disease', seed=seed, selection=selection, tweak={'fixed_psi': psi, 'totalCapacity': fixedCapacity},
                      legend=legend, ax=ax, title = title)
    meanPsis.append(mean)
    varPsis.append(var)
ax.set_title(f'Capacity = {fixedCapacity}')
fig.suptitle("Lorenz Curves for Increasing Psi Values")
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle("Mean and Variance of H for Different Psi")
ax.set_title(f'Capacity = {fixedCapacity}')

line0=ax.plot(psis, meanPsis, label='Mean H', color='red')
ax1 = ax.twinx()
line1=ax1.plot(psis, varPsis, label='Variance H', color='blue')

# added these three lines
lns = line0+line1
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)

ax.set_xlabel('Psi')
ax.set_ylabel('Mean H')
ax1.set_ylabel('Var H')
fig.show()



print(parameters_recover)


### CORRELATION:
# corr = []
# usage = results[seed]['SimpleC'].cumsum(axis = 1).to_numpy()
# time = usage.shape[1]
# for i in range(0,time):
#     corr.append(np.corrcoef(usage[:,i], results[seed]['Delta']['1'])[0][1])
#
# fig, axe = plt.subplots(1, 1, figsize=(10, 5))
# axe.scatter(x = range(101), y = corr)
# axe.set_title('Correlation between usage and Delta')
# axe.set_xlabel('Time')
# axe.set_ylabel('Correlation (Pearson)')
# fig.show()