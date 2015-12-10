from matplotlib import pyplot as plt
from matplotlib import style
from global_utils import load_phoneme_mapping
import numpy as np

def plot_error_per_epoch(err_per_epochs, legend_list, error_type):
    """
    args:
        err_per_epochs - should be a list of the different lines to plot
        legend_list - list of strings naming each line
        error_type - string containing type of error
    """
    plt.figure(facecolor='gray')
    x = np.array([i for i in range(1, len(err_per_epochs[0])+1)])
    for err in err_per_epochs:
        plt.plot(x, err)
    plt.xlabel("Epoch number")
    plt.ylabel(error_type)
    plt.xlim(1, x[-1])
    plt.legend(legend_list, loc='upper right')
    plt.show()