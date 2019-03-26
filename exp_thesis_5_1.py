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



''' ************************ Fig 5.3 Single Box example ************************ '''

exp = copy.deepcopy(g_prototype)


def x_0(x, y):
    return x - 1


def x_1(x, y):
    return - x - 1


def y_0(x, y):
    return y - 2.0


def y_1(x, y):
    return - y - 0.0 # - 1 < y


constr = [
    [x_0, x_1, y_0, y_1],
]

plot_patch = [DrawRectangle(bottom_left=(-0.5, 0.0), top_right=(0.5, 2.0))]

def constrained_region_sampler(s):
    out = ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([s, 1]))
    return out


# constrained VI
exp = copy.deepcopy(g_prototype)

exp['title'] = 'fig_5_2_constrained_VI'
exp['vi']['load_saved'] = True
exp['vi']['load_from'] = 'fig_5_2_constrained_VI_v5'

exp['data']['plt_y_domain'] = (-20.0, 6.0)
exp['vi']['run_constrained'] = True
exp['nn']['architecture'] = [1, 20, 1]


exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['constraints']['plot_between'] = []
exp['data']['integral_constrained_input_region'] = 1
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler

exp['vi']['bbb_param']['initialize_q']['mean'] = 3.0  # * torch.randn
exp['vi']['bbb_param']['initialize_q']['std'] = -10.0  # * torch.ones
exp['vi']['rv_samples'] = 1
exp['vi']['lr'] = 0.01


exp['vi']['constrained']['iterations'] = 15000
exp['vi']['constrained']['reporting_every_'] = 500
exp['vi']['constrained']['violation_samples'] = 100
exp['vi']['constrained']['gamma'] = 2000
exp['vi']['constrained']['tau_tuple'] = (15.0, 0.5)

# main_vi([exp])

# standard HMC


# constrained
exp = copy.deepcopy(g_prototype)
exp['title'] = 'fig_5_2_constrained_HMC'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_5_2_constrained_HMC_v0'

exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler
exp['data']['integral_constrained_input_region'] = 1

exp['nn']['architecture'] = [1, 20, 1]
exp['hmc']['constrained'] = True
exp['hmc']['steps'] = 10
exp['hmc']['stepsize'] = 0.005
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000
exp['hmc']['gamma'] = 2000

# main_hmc([exp])


# darting hmc

exp = copy.deepcopy(g_prototype)
exp['title'] = 'fig_5_2_darting_HMC'
exp['data']['plt_y_domain'] = (-15.0, 15.0)

exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler
exp['data']['integral_constrained_input_region'] = 1

exp['nn']['architecture'] = [1, 20, 1]

exp['hmc'] = {
    'load_saved': True,
    'load_from': 'fig_5_2_darting_HMC_v2',
    'constrained': True,
    'gamma': 5000,
    'max_violation_heuristic': True,
    'darting': {
        'bool': True,
        'preprocessing': {
            'load_saved': True,
            'load_from': 'fig_5_2_darting_HMC_v2',
            'norm_f': 0.3,
            'random_restart_scale': 3,
            'searched_modes': 30,
            'mode_searching_convergence': 0.005,
            'n_darting_regions': 6,
        },
        'algorithm': {
            'darting_region_radius': 1.5e1,
            'p_check': 0.03,
        },
    },
    'stepsize': 0.005,
    'steps': 20,
    'hmc_samples': 10000,
    'burnin': 2000,
    'thinning': 5,
}


# main_hmc([exp])


# darting hmc with added data point


X_g = torch.tensor([-2, -1.8, -1.1, 1.1, 1.8, 2]).unsqueeze(1)
Y_g = g(X_g)

X_g_prime = torch.cat([X_g, torch.tensor([[0.0]])], dim=0)
Y_g_prime = torch.cat([Y_g, torch.tensor([[4.0]])], dim=0)


exp = copy.deepcopy(g_prototype)
exp['title'] = 'fig_5_4_darting_HMC'
exp['data']['plt_y_domain'] = (-15.0, 15.0)
exp['data']['X'] = X_g_prime
exp['data']['Y'] = Y_g_prime

exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler
exp['data']['integral_constrained_input_region'] = 1

exp['nn']['architecture'] = [1, 20, 1]

exp['hmc'] = {
    'load_saved': False,
    'load_from': 'fig_5_4_darting_HMC_v3',
    'constrained': True,
    'gamma': 5000,
    'max_violation_heuristic': True,
    'darting': {
        'bool': True,
        'preprocessing': {
            'load_saved': True,
            'load_from': 'fig_5_4_darting_HMC_v3',
            'norm_f': 0.3,
            'random_restart_scale': 3,
            'searched_modes': 30,
            'mode_searching_convergence': 0.005,
            'n_darting_regions': 3,
        },
        'algorithm': {
            'darting_region_radius': 1.5e1,
            'p_check': 0.03,
        },
    },
    'stepsize': 0.005,
    'steps': 20,
    'hmc_samples': 10000,
    'burnin': 2000,
    'thinning': 5,
}


# main_hmc([exp])


# constrained nonparametric VI
exp = copy.deepcopy(g_prototype)

exp['title'] = 'fig_5_2_nonparametric_VI'
exp['vi']['alg'] = 'gumbel_softm_mog'  # 'gumbel_softm_mog' 'npv' 'npv_general'
exp['vi']['load_saved'] = True
exp['vi']['load_from'] = 'fig_5_2_nonparametric_VI_v5'

exp['data']['plt_y_domain'] = (-15.0, 17.0)
exp['vi']['run_constrained'] = True
exp['nn']['architecture'] = [1, 20, 1]


exp['constraints']['constr'] = constr
exp['constraints']['plot_patch'] = plot_patch
exp['constraints']['plot_between'] = []
exp['data']['integral_constrained_input_region'] = 1
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler

exp['vi']['gumbel_softm_mog_param'] = {
    'mixtures' : 10,
    'gumbel_tau' : 0.1,
    'reparam_estimator_samples' : 1,
    'initialize_q' : {
        'mean' : 3.0,
        'std' : -10.0
    }
}

exp['vi']['constrained']['iterations'] = 20000
exp['vi']['constrained']['reporting_every_'] = 100
exp['vi']['constrained']['violation_samples'] = 200
exp['vi']['constrained']['gamma'] = 3000
exp['vi']['constrained']['tau_tuple'] = (5.0, 0.5)


# exp['vi']['regular']['iterations'] = 10000
# exp['vi']['regular']['reporting_every_'] = 500
# exp['vi']['regular']['violation_samples'] = 100

main_vi([exp])
