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


'''
Activations
'''


class ReLUActivation(torch.autograd.Function):

    def __str__(self):
        return 'relu'

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def rbf(x): 
    return torch.exp(- x.pow(2))

relu = ReLUActivation.apply

def tanh(x): 
    return x.tanh(x)

def softrelu(x): 
    return torch.log(1.0 + x.exp())


'''
Data
'''

# f

def f(x):
    return 2 * torch.exp(- x.pow(2)) * torch.sin(5 * x)

X_f = torch.tensor([-2, -1.8, -1.5, -1, -0.8, -0.05,
                    0.05, 1.1, 1.8, 2.1]).unsqueeze(1)
Y_f = f(X_f)

X_plot_f = torch.linspace(-5, 5, steps=1000).unsqueeze(1)
Y_plot_f = f(X_plot_f)

X_id_f = torch.tensor([-2.1, -1.6, -0.8, -0.3, 0.5, 1.6, 1.8]).unsqueeze(1)
Y_id_f = f(X_id_f)

X_ood_f = torch.tensor([-4, -3.1, -2.4, 2.7, 3, 4.4]).unsqueeze(1)
Y_ood_f = f(X_ood_f)

# g

def g(x):
    return - 0.6666 * x.pow(4) + 4/3 * x.pow(2) + 1


X_g = torch.tensor([-2, -1.8, -1.1, 1.1, 1.8, 2]).unsqueeze(1)
Y_g = g(X_g)

X_plot_g = torch.linspace(-5, 5, steps=1000).unsqueeze(1)
Y_plot_g = g(X_plot_g)

X_id_g = torch.tensor([-1.9, -1.5, 0.5, 0.0, 0.5, 1.5, 1.9]).unsqueeze(1)
Y_id_g = g(X_id_g)

X_ood_g = torch.tensor([-4, -3, -2.5, 2.5, 3, 4]).unsqueeze(1)
Y_ood_g = g(X_ood_g)


# prototypes

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
        'plt_size': (5, 4),
        'integral_constrained_input_region': 0,
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
        'plot_patch': [],
        'plot_between': [],
    },
    'vi': {  # alg options: bbb, npv
        'alg': 'bbb',
        'run_constrained' : False,
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
    }
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
        'plot_patch': [],
        'plot_between': [],
    },
    'vi': {  # alg options: bbb, npv
        'alg': 'bbb',
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
