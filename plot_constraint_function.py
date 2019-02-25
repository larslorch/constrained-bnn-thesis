import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


matplotlib.pyplot.rc('font', family='serif')

import math
from pprint import pprint
# from matplotlib import rc
import torch
from utils import *
from plot import *

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def y_c00(x, y): 
    return y + 1 + 2 *  torch.exp(- 0.5 * x.pow(2))

def x_c10(x, y): 
    return x.pow(2) - y

constr = [
    [y_c00],
    [x_c10]
]


x_dim, y_dim = 500, 500
x_ticks = 5
x_lim = (-4, 4)
y_ticks = 4
y_lim = (-5, 10)
tau_s, tau_b = 0.5, 5.0


# Compute constraint function c_S
xs = torch.linspace(x_lim[0], x_lim[1], steps=x_dim).repeat(y_dim, 1).transpose(0, 1)
ys = torch.linspace(y_lim[1], y_lim[0], steps=y_dim).repeat(x_dim, 1)
c = torch.zeros(xs.shape)
for region in constr:
    d = torch.ones(xs.shape)
    for constraint in region:
        d *= psi(constraint(xs, ys), tau_s, tau_b)
    c += d

# 2D Heatmap of M where M[x, y] is plotted at (x, y)
def plot_2D_heatmap(M, x_ticks=10, x_lim=(-5, 5), y_ticks=10, y_lim=(-5, 5)):

    M = np.transpose(M)

    xticks_loc = np.linspace(0, M.shape[1], num=x_ticks, endpoint=True, dtype=np.int)
    xticks_arr = np.linspace(x_lim[0], x_lim[1], x_ticks, dtype=np.int)

    yticks_loc = np.linspace(0, M.shape[0], num=y_ticks, endpoint=True, dtype=np.int)
    yticks_arr = np.linspace(y_lim[0], y_lim[1], y_ticks, dtype=np.int)

    ax = sns.heatmap(M, linewidth=0, xticklabels=xticks_arr,
                     yticklabels=yticks_arr[::-1])

    ax.set_xticks(xticks_loc)
    ax.set_yticks(yticks_loc)
    plt.xticks(rotation=0)
    
    fig = plt.gcf()  # get current figure
    fig.set_size_inches((4.5, 3.5))

    plt.savefig('experiment_results/plot_constraint_function.png',
                format='png', frameon=False, dpi=400)

    # plt.show()


plot_2D_heatmap(c, 
    x_ticks=x_ticks, x_lim=x_lim,
    y_ticks=y_ticks, y_lim=y_lim)