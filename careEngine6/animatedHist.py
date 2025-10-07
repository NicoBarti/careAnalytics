import pandas as pd
import os  # import os module
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

def collector(directory):
    means = pd.DataFrame({})
    for entry in os.scandir(directory):
        if entry.is_file() and "csv" in entry.name.split("."):  # check if it's a file
            dd = pd.read_csv(entry.path)
            means = pd.concat([means, dd], axis=0)
    return (means)

def plotOutput(data, name, onlyMeans=False, finalplot = False):
    Fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
    simulationsCounter = data.shape[0]+1
    if(not onlyMeans):
        for row in range(data.shape[0]):
            axe.scatter(x=np.arange(start=1, stop=data.shape[1]+1, step=1), y=data.iloc[row,:], alpha=0.05, color="blue")
    axe.set_title(f"{simulationsCounter} repetitions")
    axe.set_xlabel("Time steps (weeks)")
    axe.set_ylabel(name)
    return(Fig)

def mobileHist(data):

    def prepare_animation(bar_container):
        def animate(frame_number):
            # simulate new data coming in
            new_data = data[str(frame_number)]
            n, _ = np.histogram(new_data, HIST_BINS)
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
            return bar_container.patches

        return animate

    HIST_BINS = np.linspace(0, 20, 100)
    n, _ = np.histogram(data, HIST_BINS)

    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(data, HIST_BINS, lw=1,
                                  ec="yellow", fc="green", alpha=0.5)
    ax.set_ylim(top=55)  # set safe limit to ensure that all data is visible.

    ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 50,
                                  repeat=False, blit=True)
    plt.show()

    #fig, axe = plt.subplots(1, 1, figsize=(10, 10), facecolor='ghostwhite')
    #windows = np.arange(start=1, stop=500, step=30)
    #artists = []
    #for i in windows:
    #    container = axe.hist([x if x < 5 else 5 for x in data[str(i)]], range = (0,5))
    #    artists.append(container)
    #ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    #plt.show()

data = collector("../data/sensitivity_1/H")
#mobileHist(dataH)



# Fixing bin edges
HIST_BINS = np.linspace(0, 5, 100)

# histogram our data with numpy
#data = np.random.randn(1000)
n, _ = np.histogram(data["1"], HIST_BINS)
print(1)

def prepare_animation(bar_container):

    def animate(frame_number):
        # update histogram
        #newdata = np.random.randn(1000)
        #newdata = data[str(frame_number+1)].to_numpy()
        newdata = np.array([x if x < 5 else 5 for x in data[str(frame_number+1)]])
        n, _ = np.histogram(newdata, HIST_BINS)
        for count, rect in zip(n, bar_container.patches):
            rect.set_height(count)
        return bar_container.patches
    return animate

# Output generated via `matplotlib.animation.Animation.to_jshtml`.
fig, ax = plt.subplots()
_, _, bar_container = ax.hist(data["1"], HIST_BINS, lw=1,
                              ec="yellow", fc="green", alpha=0.5)
ax.set_ylim(top=300)  # set safe limit to ensure that all data is visible.

ani = animation.FuncAnimation(fig, prepare_animation(bar_container),  frames=499, interval=30, blit=True, repeat = False)
#animation.PillowWriter(ani)
plt.show()
ani.save(filename="/Users/nicolasbarticevic/Desktop/CareEngineAnalytics/figures/anim1.apng", writer="pillow")
