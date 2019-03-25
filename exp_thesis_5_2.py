import matplotlib.pyplot as plt
import math
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

import copy

from utils import *
from plot import *
from main_vi import main_vi
from main_hmc import main_hmc

from plot_CONST import *



'''
Constraints: f(x, y) <= 0

Have to be broadcastable!!
i.e. x is (datapoints, in_dim)
     y is (weights samples, datapoints, out_dim)


'''

# DON'T FORGET:
#   1)  Double check constraint def. (define _infeasible_ regions)
#   2)  constaint sampler
#   3)  plot rectangle



'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''



''' ************************ Fig 5.5 Multi Box example ************************ '''

exp = copy.deepcopy(g_prototype)

# right
def y_1_0(x, y):
    return 3 * y.pow(2) - x + 3 # x - 3 > 3y^2

# left
def y_2_0(x, y):
    return 3 * y.pow(2) + x + 3   # x + 3 < -3y^2


def x_3_0(x, y):
    return x - 0.5  # x < 0.5


def x_3_1(x, y):
    return - x - 0.5 # x > -0.5


def y_3_0(x, y):
    return y - 2.0 # y < 2


def y_3_1(x, y):
    return - y  # 0 < y


def x_4_0(x, y):
    return x - 1.5 # x < 1.5


def x_4_1(x, y):
    return - x - 1.5  # x > -1.5


def y_4_0(x, y):
    return y + x.pow(4) + 5 # y < - x^4 - 5


constr = [
    [y_1_0],
    [y_2_0],
    [x_3_0, x_3_1, y_3_0, y_3_1],
    [x_4_0, x_4_1, y_4_0],
]

plot_patch = [DrawRectangle(bottom_left=(-0.5, 0.0), top_right=(0.5, 2.0))]


X_plot_1 = torch.linspace(3.001, 6, steps=1000)
X_plot_2 = torch.linspace(-6, -3.001, steps=1000)
X_plot_4 = torch.linspace(-1.5, 1.5, steps=1000)

plot_between = [
    (X_plot_1, - (X_plot_1 / 3 - 1).pow(0.5), (X_plot_1 / 3 - 1).pow(0.5)),
    (X_plot_2, - (- X_plot_2 / 3 - 1).pow(0.5), (- X_plot_2 / 3 - 1).pow(0.5)),
    (X_plot_4, -15 * torch.ones(X_plot_4.shape), - X_plot_4.pow(4) - 5),
]


def constrained_region_sampler(s):
    out_1 = ds.Uniform(3, 6).sample(sample_shape=torch.Size([s, 1]))
    out_2 = ds.Uniform(-6, -3).sample(sample_shape=torch.Size([s, 1]))
    out_3 = ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([s, 1]))
    out_4 = ds.Uniform(-1.5, 1.5).sample(sample_shape=torch.Size([s, 1]))
    out = torch.cat([out_1, out_2, out_3, out_4], dim=0)
    return out


# darting hmc

exp = copy.deepcopy(g_prototype)
exp['title'] = 'fig_5_5_darting_HMC'
exp['data']['plt_y_domain'] = (-15.0, 15.0)

exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['constraints']['plot_between'] = plot_between
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler
exp['data']['integral_constrained_input_region'] = 12

exp['nn']['architecture'] = [1, 20, 1]

exp['hmc'] = {
    'load_saved': False,
    'load_from': '',
    'constrained': True,
    'gamma': 5000,
    'max_violation_heuristic': True,
    'darting': {
        'bool': True,
        'preprocessing': {
            'load_saved': True,
            'load_from': 'fig_5_5_darting_HMC_v0',
            'norm_f': 0.3,
            'random_restart_scale': 3,
            'searched_modes': 50,
            'mode_searching_convergence': 0.005,
            'n_darting_regions': 10,
        },
        'algorithm': {
            'darting_region_radius': 1.5e1,
            'p_check': 0.03,
        },
    },
    'stepsize': 0.005,
    'steps': 20,
    'hmc_samples': 4100,
    'burnin': 100,
    'thinning': 5,
}


main_hmc([exp])

