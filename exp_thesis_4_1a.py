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


''' ************************ Fig 4.1 ground truth f ************************ '''

exp = copy.deepcopy(f_prototype)

exp['title'] = 'fig_4_1'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_1_v1'

# main_hmc([exp])


''' ************************ Fig 4.2 ground truth g ************************ '''
exp = copy.deepcopy(g_prototype)

exp['title'] = 'fig_4_2'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_2_v1'

# main_hmc([exp])

''' ************************ Fig 4.3 linear bound ************************ '''
exp = copy.deepcopy(f_prototype)

def y_4_3_0(x, y):
    return y + 2.5

def y_4_3_1(x, y):
    return - y + 2.5

constr = [
    [y_4_3_0],
    [y_4_3_1],
]

plot_patch = [
    DrawRectangle(bottom_left=(-6, -20), top_right=(6, -2)),
    DrawRectangle(bottom_left=(-6, 2), top_right=(6, 20)),
]

def constrained_region_sampler_4_3(s):
    out = ds.Uniform(-6, 6).sample(sample_shape=torch.Size([s, 1]))
    return out


exp['title'] = 'fig_4_3'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_3_v0'


exp['data']['plt_y_domain'] = (-4.0, 4.0)
exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12

exp['hmc']['constrained'] = True
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000
exp['hmc']['gamma'] = 5000

# main_hmc([exp])

''' ************************ Fig 4.3 nonlinear bound ************************ '''
exp = copy.deepcopy(f_prototype)

def y_4_3_2(x, y):
    return y + 0.5 + 2 * torch.exp(- x.pow(2))


def y_4_3_3(x, y):
    return - y + 0.5 + 2 * torch.exp(- x.pow(2))


constr_4_3_2 = [
    [y_4_3_2],
    [y_4_3_3],
]

plot_patch = []

plot_between = [
    (X_plot_f, 0.5 + 2 * torch.exp(- X_plot_f.pow(2)), 10 * torch.ones(X_plot_f.shape)),
    (X_plot_f, -10 * torch.ones(X_plot_f.shape), -0.5 - 2 * torch.exp(- X_plot_f.pow(2))),
]


exp['title'] = 'fig_4_4'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_4_v4'

exp['data']['plt_y_domain'] = (-4.0, 4.0)

exp['constraints']['constr'] = constr_4_3_2
exp['constraints']['plot_patch'] = plot_patch
exp['constraints']['plot_between'] = plot_between
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12


exp['hmc']['constrained'] = True
exp['hmc']['steps'] = 20
exp['hmc']['stepsize'] = 0.005
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000
exp['hmc']['gamma'] = 2000


# main_hmc([exp])

''' ************************ Table 4.1 nonlinear bound ************************ '''

''' 1 Layer - 20 nodes '''

# standard
exp = copy.deepcopy(f_prototype)

exp['title'] = 'fig_4_1'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_1_v1' # same as baseline

exp['constraints']['constr'] = constr_4_3_2
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12


# main_hmc([exp])

''' 1 Layer - 50 nodes '''

# standard
exp = copy.deepcopy(f_prototype)
exp['title'] = 'tab_4_1_50_no'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'tab_4_1_50_no_v0'

exp['constraints']['constr'] = constr_4_3_2
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12

exp['nn']['architecture'] = [1, 50, 1]
exp['hmc']['constrained'] = False
exp['hmc']['steps'] = 20
exp['hmc']['stepsize'] = 0.005
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000
exp['hmc']['gamma'] = 2000

# main_hmc([exp])

# constrained
exp = copy.deepcopy(f_prototype)
exp['title'] = 'tab_4_1_50_yes'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'tab_4_1_50_yes_v0'

exp['constraints']['constr'] = constr_4_3_2
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12

exp['nn']['architecture'] = [1, 50, 1]
exp['hmc']['constrained'] = True
exp['hmc']['steps'] = 10
exp['hmc']['stepsize'] = 0.005
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000
exp['hmc']['gamma'] = 2000

# main_hmc([exp])

''' 2 Layer - 50 nodes '''

# standard
exp = copy.deepcopy(f_prototype)
exp['title'] = 'tab_4_2_50_no'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'tab_4_2_50_no_v3'

exp['constraints']['constr'] = constr_4_3_2
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12

exp['nn']['architecture'] = [1, 50, 50, 1]
exp['hmc']['constrained'] = False
exp['hmc']['steps'] = 20
exp['hmc']['stepsize'] = 0.002
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000
exp['hmc']['gamma'] = 2000

# main_hmc([exp])

# constrained
exp = copy.deepcopy(f_prototype)

exp['title'] = 'tab_4_2_50_yes'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'tab_4_2_50_yes'

exp['constraints']['constr'] = constr_4_3_2
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12

exp['nn']['architecture'] = [1, 50, 50, 1]
exp['hmc']['constrained'] = True
exp['hmc']['steps'] = 10
exp['hmc']['stepsize'] = 0.002
exp['hmc']['hmc_samples'] = 3000
exp['hmc']['burnin'] = 1000
exp['hmc']['gamma'] = 2000

# main_hmc([exp])


