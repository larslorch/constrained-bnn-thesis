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



''' ************************ Fig 4.5 equality constraint ************************ '''

exp = copy.deepcopy(g_prototype)

epsilon = 0.125

def x_4_5_0(x, y):
    return x - epsilon

def x_4_5_1(x, y):
    return -x - epsilon

def y_4_5_0(x, y):
    return y - 1 + epsilon / 2


def y_4_5_1(x, y):
    return - y + 1 - epsilon / 2


constr_4_5 = [
    [x_4_5_0, x_4_5_1, y_4_5_0],
    [x_4_5_0, x_4_5_1, y_4_5_1],
]

plot_patch = [
    DrawRectangle(bottom_left=(-epsilon, -20), top_right=(epsilon, 1 - epsilon)),
    DrawRectangle(bottom_left=(-epsilon, 1 + epsilon), top_right=(epsilon, 20)),
]


plot_between = []


def constrained_region_sampler_4_5(s):
    # out = torch.cat([
    #     ds.Uniform(-6, -1).sample(sample_shape=torch.Size([s, 1])),
    #     ds.Uniform(1, 6).sample(sample_shape=torch.Size([s, 1]))
    # ], dim=0)
    out = ds.Uniform(-epsilon, epsilon).sample(sample_shape=torch.Size([s, 1]))
    return out


exp['title'] = 'fig_4_5'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_5_v2'


exp['data']['plt_y_domain'] = (-11.0, 4.0)
exp['nn']['architecture'] = [1, 20, 1]

exp['constraints']['plot_patch'] = plot_patch
exp['constraints']['plot_between'] = []

exp['constraints']['constr'] = constr_4_5
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_5
exp['data']['integral_constrained_input_region'] = 2 * epsilon

exp['hmc']['constrained'] = True
exp['hmc']['steps'] = 35
exp['hmc']['stepsize'] = 0.0005
exp['hmc']['hmc_samples'] = 10000
exp['hmc']['burnin'] = 1000
exp['hmc']['gamma'] = 20000


main_hmc([exp])


# unconstrained with extra point

exp['title'] = 'fig_4_5_extra'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_5_extra_v2'

exp['data']['plt_y_domain'] = (-12.0, 7.0)
exp['nn']['architecture'] = [1, 20, 1]

exp['constraints']['plot_patch'] = []
exp['constraints']['plot_between'] = []

exp['constraints']['constr'] = constr_4_5
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_5
exp['data']['integral_constrained_input_region'] = 2 * epsilon

exp['hmc']['constrained'] = False
exp['hmc']['steps'] = 25
exp['hmc']['stepsize'] = 0.005
exp['hmc']['hmc_samples'] = 3000
exp['hmc']['burnin'] = 1000
exp['hmc']['gamma'] = 20000


# main_hmc([exp])
