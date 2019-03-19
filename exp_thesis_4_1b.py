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



''' ************************ Fig 4.4 asymptotic bound (VI) ************************ '''

# standard bbb as benchmark

exp = copy.deepcopy(f_prototype)


def x_4_4_0(x, y):
    return x + 1

def x_4_4_1(x, y):
    return -x + 1

def y_4_4_0(x, y):
    return y + 0.5 + 2 * torch.exp(- x.pow(2))


def y_4_4_1(x, y):
    return - y + 0.5 + 2 * torch.exp(- x.pow(2))


constr_4_4 = [
    [x_4_4_0, y_4_4_0],
    [x_4_4_0, y_4_4_1],
    [x_4_4_1, y_4_4_0],
    [x_4_4_1, y_4_4_1],
]

plot_patch = []

X_plot_4_4a = torch.linspace(-6, -1, steps=500) 
X_plot_4_4b = torch.linspace(1, 6, steps=500)

plot_between = [
    (X_plot_4_4a, 0.5 + 2 * torch.exp(- X_plot_4_4a.pow(2)),
     10 * torch.ones(X_plot_4_4a.shape)),
    (X_plot_4_4a, -10 * torch.ones(X_plot_4_4a.shape), -
     0.5 - 2 * torch.exp(- X_plot_4_4a.pow(2))),
    (X_plot_4_4b, 0.5 + 2 * torch.exp(- X_plot_4_4b.pow(2)),
     10 * torch.ones(X_plot_4_4b.shape)),
    (X_plot_4_4b, -10 * torch.ones(X_plot_4_4b.shape), -
     0.5 - 2 * torch.exp(- X_plot_4_4b.pow(2))),
]


def constrained_region_sampler_4_4(s):
    out = torch.cat([
        ds.Uniform(-6, -1).sample(sample_shape=torch.Size([s, 1])),
        ds.Uniform(1, 6).sample(sample_shape=torch.Size([s, 1]))
    ], dim=0)
    return out


exp['title'] = 'fig_4_4'
exp['vi']['load_saved'] = True
exp['vi']['load_from'] = 'fig_4_4_v16'


exp['data']['plt_y_domain'] = (-5.0, 5.0)
exp['vi']['run_constrained'] = False
exp['nn']['architecture'] = [1, 50, 1]


exp['constraints']['constr'] = constr_4_4
exp['constraints']['plot_patch'] = plot_patch
exp['constraints']['plot_between'] = plot_between
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_4
exp['data']['integral_constrained_input_region'] = 10

exp['vi']['bbb_param']['initialize_q']['mean'] = 5.0  # * torch.randn
exp['vi']['bbb_param']['initialize_q']['std'] = -10.0  # * torch.ones
exp['vi']['rv_samples'] = 1
exp['vi']['lr'] = 0.005
exp['vi']['regular'] =  {
    'iterations': 350,
    'restarts': 1,
    'reporting_every_': 500,
    'cores_used': 1,
}

main_vi([exp])
