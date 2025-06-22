# quick graph for accuracy latency energy metric
# plots a graph of accuracy*latency vs energy 

import matplotlib.pyplot as plt
import numpy as np


acc_dvts = [42, 50.8, 53.4, 55.4]
acc_bon = [40.4, 45.4, 47.2, 49.6]


E_bon = [2.184, 3.228, 5.8, 13.962]
lat_bon = [840.144, 3360.576, 13442.306, 53769.223]
 
E_dvts = [3.702, 7.123, 19.01, 82.846]
lat_dvts = [4816.642, 19266.567, 77066.267, 308265.066]

def plot_accLat_vs_energy(
    acc1, lat1, energy1, 
    acc2, lat2, energy2, 
    label1='Series 1', label2='Series 2',
    x_limits=None, y_limits=None
):
    """
    Plots two Accuracy*Latency vs Energy curves on the same graph.
    
    acc1, lat1, energy1  – lists for the first series (e.g. 'dvts')
    acc2, lat2, energy2  – lists for the second series (e.g. 'bon')
    label1, label2       – legend labels for each series
    x_limits             – tuple (xmin, xmax) or None
    y_limits             – tuple (ymin, ymax) or None
    """
    # compute Accuracy * Latency for each series
    acc_lat1 = [a * l for a, l in zip(acc1, lat1)]
    acc_lat2 = [a * l for a, l in zip(acc2, lat2)]
    
    # (optionally) sort each series by acc_lat so the lines don't zigzag
    pairs1 = sorted(zip(energy1, acc_lat1))
    pairs2 = sorted(zip(energy2, acc_lat2))
    xs1, ys1 = zip(*pairs1)
    xs2, ys2 = zip(*pairs2)
    
    # plot both series
    plt.plot(xs1, ys1, marker='o', linestyle='-', label=label1)
    plt.plot(xs2, ys2, marker='s', linestyle='--', label=label2)
    
    # set axis limits if given
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
       plt.ylim(y_limits)
    
    # labels, title, legend, grid
    plt.ylabel('Accuracy-Latency Product')
    plt.xlabel('Energy (kWh)')
    plt.title('Llama 3.2 1B Accuracy-Latency vs Energy')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
    # Show the plot
    plt.show()
if __name__=="__main__":

    plot_accLat_vs_energy(
    acc_dvts, lat_dvts, E_dvts,
    acc_bon,  lat_bon,  E_bon,
    label1='dvts', label2='bon',
    x_limits=(0, 15), y_limits=(0, 0.3e7)
)