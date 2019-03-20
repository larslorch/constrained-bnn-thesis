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



''' ************************ Fig 4.8 Ablation Experiment: tau ************************ '''

exp = copy.deepcopy(f_prototype)

def y_0(x, y):
    return y + 0.5 + 2 * torch.exp(- x.pow(2))


def y_1(x, y):
    return - y + 0.5 + 2 * torch.exp(- x.pow(2))


constr = [
    [y_0],
    [y_1]
]

plot_patch = []

X_plot = torch.linspace(-6, 6, steps = 1000)
plot_between = [
    (X_plot, 0.5 + 2 * torch.exp(- X_plot.pow(2)),
     15 * torch.ones(X_plot.shape)),
    (X_plot, -15 * torch.ones(X_plot.shape), -
     0.5 - 2 * torch.exp(- X_plot.pow(2))),
]


def constrained_region_sampler(s):
    out = ds.Uniform(-6, 6).sample(sample_shape=torch.Size([s, 1]))
    return out


# constrained: tau 3, 1
exp_gam = copy.deepcopy(f_prototype)


exp_gam['title'] = 'fig_4_8_tau_3_1'
exp_gam['vi']['load_saved'] = False
exp_gam['vi']['load_from'] = ''


exp_gam['data']['plt_y_domain'] = (-10.0, 10.0)
exp_gam['vi']['run_constrained'] = True
exp_gam['nn']['architecture'] = [1, 20, 1]


exp_gam['constraints']['constr'] = constr
exp_gam['constraints']['plot_patch'] = []
exp_gam['constraints']['plot_between'] = plot_between
exp_gam['data']['integral_constrained_input_region'] = 12

exp_gam['vi']['bbb_param']['initialize_q']['mean'] = 3.0  # * torch.randn
exp_gam['vi']['bbb_param']['initialize_q']['std'] = -10.0  # * torch.ones
exp_gam['vi']['rv_samples'] = 1
exp_gam['vi']['lr'] = 0.01

exp_gam['vi']['constrained'] = {
    'iterations': 10000,
    'restarts': 1,
    'reporting_every_': 100,
    'violation_samples': 300,
    'tau_tuple': (3.0, 1.0),
    'gamma': 1000,
    'constrained_region_sampler': constrained_region_sampler,
}

main_vi([exp_gam])

# constrained: tau 15 1
exp = copy.deepcopy(exp_gam)

exp['title'] = 'fig_4_8_tau_15_1'
exp['vi']['load_saved'] = False
exp['vi']['load_from'] = ''

exp['vi']['lr'] = 0.01
exp['vi']['constrained']['iterations'] = 10000
exp['vi']['constrained']['reporting_every_'] = 100
exp['vi']['constrained']['violation_samples'] = 300
exp['vi']['constrained']['gamma'] = 1000
exp['vi']['constrained']['tau_tuple'] = (15.0, 1.0)

main_vi([exp])


# constrained: tau 3, 0.5
exp = copy.deepcopy(exp_gam)

exp['title'] = 'fig_4_8_tau_3_05'
exp['vi']['load_saved'] = False
exp['vi']['load_from'] = ''

exp['vi']['lr'] = 0.01
exp['vi']['constrained']['iterations'] = 10000
exp['vi']['constrained']['reporting_every_'] = 100
exp['vi']['constrained']['violation_samples'] = 300
exp['vi']['constrained']['gamma'] = 1000
exp['vi']['constrained']['tau_tuple'] = (3.0, 0.5)

main_vi([exp])


# constrained: tau 15, 0.5
exp = copy.deepcopy(exp_gam)

exp['title'] = 'fig_4_8_tau_15_05'
exp['vi']['load_saved'] = False
exp['vi']['load_from'] = ''

exp['vi']['lr'] = 0.01
exp['vi']['constrained']['iterations'] = 10000
exp['vi']['constrained']['reporting_every_'] = 100
exp['vi']['constrained']['violation_samples'] = 300
exp['vi']['constrained']['gamma'] = 1000
exp['vi']['constrained']['tau_tuple'] = (15.0, 0.5)

main_vi([exp])
