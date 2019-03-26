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


''' ************************ Fig 4.9 contradicting observations ************************ '''

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
    DrawRectangle(bottom_left=(-6, -20), top_right=(6, -2.5)),
    DrawRectangle(bottom_left=(-6, 2.5), top_right=(6, 20)),
]

def constrained_region_sampler_4_3(s):
    out = torch.cat([
        ds.Uniform(-6, 6).sample(sample_shape=torch.Size([s, 1])),
        ds.Uniform(-1, 0).sample(sample_shape=torch.Size([s, 1]))
    ], dim=0)

    return out


X_f = torch.tensor([-2, -1.8, -1.5, -1, -0.8, -0.05,
                    0.05, 1.1, 1.8, 2.1]).unsqueeze(1)
Y_f = f(X_f)

X_f_prime = torch.cat([X_f, torch.tensor([[-0.5]])], dim=0)
Y_f_prime = torch.cat([Y_f, torch.tensor([[4.0]])], dim=0)





exp['title'] = 'fig_4_9'
exp['vi']['load_saved'] = False
exp['vi']['load_from'] = 'fig_4_9_v1'

exp['data']['X'] = X_f_prime
exp['data']['Y'] = Y_f_prime

exp['data']['plt_y_domain'] = (-4.0, 5.0)
exp['vi']['run_constrained'] = True
exp['nn']['architecture'] = [1, 20, 1]


exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3
exp['data']['integral_constrained_input_region'] = 12

exp['vi']['bbb_param']['initialize_q']['mean'] = 3.0  # * torch.randn
exp['vi']['bbb_param']['initialize_q']['std'] = -10.0  # * torch.ones
exp['vi']['rv_samples'] = 1
exp['vi']['lr'] = 0.01

exp['vi']['constrained'] = {
    'iterations': 20000,
    'restarts': 1,
    'reporting_every_': 500,
    'violation_samples': 300,
    'tau_tuple': (25.0, 10.0),
    'gamma': 10000,
    'constrained_region_sampler': constrained_region_sampler_4_3,
}

main_vi([exp])
