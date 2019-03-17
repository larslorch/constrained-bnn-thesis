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
all_experiments = []


f_prototype = {
    'title': 'thesis_uni',
    'nn': {
        'architecture': [1, 20, 1],
        'nonlinearity': rbf,
        'prior_ds': ds.Normal(0.0, 3.5),
    },
    'data': {
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-4.0, 4.0),
        'plt_y_domain': (-9, 14.0),
        'plt_size' : (5, 4),
        'integral_constrained_region': 0,
        'X':  X_f,
        'Y':  Y_f,
        'X_plot': X_plot_f,
        'Y_plot': Y_plot_f,
        'X_v_id': X_id_f,
        'Y_v_id': Y_id_f,
        'X_v_ood': X_ood_f,
        'Y_v_ood': Y_ood_f,
    },
    'constraints': {
        'constr': [],
        'plot': [],
    },
    'vi': { # alg options: bbb, npv
        'alg': 'npv',
        'bbb_param' : {
            'initialize_q': {
                'mean': 1.0,  # * torch.randn
                'std': -2.5,  # * torch.ones
            },
        },
        'npv_param': {
            'mixtures' : 10, 
            'initialize_q': {
                'mean': 1.0,  # * torch.randn
                'std': 0.0,  # * torch.ones
            },
        },
        'rv_samples': 100,
        'batch_size': 0,  # batch_size = 0 implies full dataset training
        'lr' : 0.01,
        'regular': {
            'iterations': 200,
            'restarts': 1,
            'reporting_every_': 10,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 200,
            'restarts': 1,
            'reporting_every_': 10,
            'cores_used': 1,
            'violation_samples': 5000,
            'tau_tuple': (15.0, 2.0),
            'gamma': 1000,
            'constrained_region_sampler': None,
        },
        'posterior_predictive_analysis': {
            'posterior_samples': 50,
            'constrained_region_samples_for_pp_violation': 50,
        }
    },
    'hmc': {
        'load_saved' : False,
        'load_from': 'thesis_uni_v999',
        'constrained' : False,
        'gamma' : 5000,
        'max_violation_heuristic' : False,
        'darting': {
            'bool': False,
            'preprocessing': {
                'load_saved': False,
                'load_from': 'thesis_uni_v999',
                'norm_f': 0.3,
                'random_restart_scale' : 3,
                'searched_modes': 20,
                'mode_searching_convergence': 0.01,
                'n_darting_regions': 3,
            },
            'algorithm': {
                'darting_region_radius': 2.0e1,
                'p_check': 0.03,
            },
        },
        'stepsize': 0.005,
        'steps': 25,
        'hmc_samples': 7000,
        'burnin': 2000,
        'thinning': 3,
    },
    'experiment': {
        'run_regular_vi': True,
        'run_constrained_vi': False,
        'multithread_computation': False,
        'compute_held_out_loglik_id': True,
        'compute_held_out_loglik_ood': False,
        'compute_RMSE_id': True,
        'compute_RMSE_ood': False,
        'show_function_samples': True,
        'show_posterior_predictive': True,
        'show_posterior_predictive_ind': (True, 500),
        'show_plot_training_evaluations': True,
        'show_constraint_function_heatmap': False,
        'plot_size': (6, 4),  # width, height (inches)
    },
}

g_prototype = {
    'title': 'thesis_uni',
    'nn': {
        'architecture': [1, 20, 1],
        'nonlinearity': rbf,
        'prior_ds': ds.Normal(0.0, 3.5),
    },
    'data': {
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-5, 5),
        'plt_y_domain': (-12, 12),
        'plt_size': (5, 4),
        'integral_constrained_region': 0,
        'X':  X_g,
        'Y':  Y_g,
        'X_plot': X_plot_g,
        'Y_plot': Y_plot_g,
        'X_v_id': X_id_g,
        'Y_v_id': Y_id_g,
        'X_v_ood': X_ood_g,
        'Y_v_ood': Y_ood_g,
    },
    'constraints': {
        'constr': [],
        'plot': [],
    },
    'vi': {  # alg options: bbb, npv
        'alg': 'npv',
        'bbb_param': {
            'initialize_q': {
                'mean': 1.0,  # * torch.randn
                'std': -2.5,  # * torch.ones
            },
        },
        'npv_param': {
            'mixtures': 10,
            'initialize_q': {
                'mean': 1.0,  # * torch.randn
                'std': 0.0,  # * torch.ones
            },
        },
        'rv_samples': 100,
        'batch_size': 0,  # batch_size = 0 implies full dataset training
        'lr': 0.01,
        'regular': {
            'iterations': 200,
            'restarts': 1,
            'reporting_every_': 10,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 200,
            'restarts': 1,
            'reporting_every_': 10,
            'cores_used': 1,
            'violation_samples': 5000,
            'tau_tuple': (15.0, 2.0),
            'gamma': 1000,
            'constrained_region_sampler': None,
        },
        'posterior_predictive_analysis': {
            'posterior_samples': 50,
            'constrained_region_samples_for_pp_violation': 50,
        }

    },
    'hmc': {
        'load_saved': False,
        'load_from': 'thesis_uni_v999',
        'constrained': False,
        'gamma': 5000,
        'max_violation_heuristic': False,
        'darting': {
            'bool': False,
            'preprocessing': {
                'load_saved': False,
                'load_from': 'thesis_uni_v999',
                'norm_f': 0.3,
                'random_restart_scale': 3,
                'searched_modes': 20,
                'mode_searching_convergence': 0.01,
                'n_darting_regions': 3,
            },
            'algorithm': {
                'darting_region_radius': 2.0e1,
                'p_check': 0.03,
            },
        },
        'stepsize': 0.005,
        'steps': 25,
        'hmc_samples': 7000,
        'burnin': 2000,
        'thinning': 5,
    },
    'experiment': {
        'run_regular_vi': True,
        'run_constrained_vi': False,
        'multithread_computation': False,
        'compute_held_out_loglik_id': True,
        'compute_held_out_loglik_ood': False,
        'compute_RMSE_id': True,
        'compute_RMSE_ood': False,
        'show_function_samples': True,
        'show_posterior_predictive': True,
        'show_posterior_predictive_ind': (True, 500),
        'show_plot_training_evaluations': True,
        'show_constraint_function_heatmap': False,
        'plot_size': (6, 4),  # width, height (inches)
    },
}




'''Fig 4.1 ground truth f'''

exp = copy.deepcopy(f_prototype)

exp['title'] = 'fig_4_1'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_1_v1'

# main_hmc([exp])


'''Fig 4.2 ground truth g'''
exp = copy.deepcopy(g_prototype)

exp['title'] = 'fig_4_2'
exp['hmc']['load_saved'] = True
exp['hmc']['load_from'] = 'fig_4_2_v1'

# main_hmc([exp])

'''Fig 4.3 linear bound'''
exp = copy.deepcopy(f_prototype)

def y_4_3_0(x, y):
    return y + 2

def y_4_3_1(x, y):
    return - y + 2

constr = [
    [y_4_3_0],
    [y_4_3_1],
]

plot_constr = [
    DrawRectangle(bottom_left=(-6, -20), top_right=(6, -2)),
    DrawRectangle(bottom_left=(-6, 2), top_right=(6, 20)),
]

def constrained_region_sampler_4_3(s):
    out = ds.Uniform(-6, 6).sample(sample_shape=torch.Size([s, 1]))
    return out


exp['title'] = 'fig_4_3'

exp['data']['plt_y_domain'] = (-4.0, 40)
exp['constraints']['constr'] = constr
exp['constraints']['plot'] = plot_constr
exp['vi']['constrained']['constrained_region_sampler'] = constrained_region_sampler_4_3

exp['hmc']['constrained'] = True
exp['hmc']['hmc_samples'] = 7000
exp['hmc']['burnin'] = 2000

main_hmc([exp])
