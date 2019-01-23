import matplotlib.pyplot as plt
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable

import copy



'''
Preliminary definitions
'''


def rbf(x): return torch.exp(- x.pow(2))
def relu(x): return x.clamp(min=0.0)
def tanh(x): return x.tanh(x)

N, n_dim = 10, 1

def ground_truth(x):
    return x.pow(2)
    

X =  torch.randn(20, 1)
Y = ground_truth(X) + ds.Normal(0.0, 0.1).sample(X.shape)

X_plot = torch.linspace(-5, 5, steps=100).unsqueeze(1)
X_id = torch.randn(20, 1)
X_ood = torch.randn(20, 1) + 3


'''
Constraints

Have to be broadcastable!!
i.e. x is (datapoints, in_dim)
     y is (weights samples, datapoints, out_dim)


'''

# for all x: 
#  0 < y
def y_c0(x, y):
    return - y

constr = [
    ([], [y_c0]),
]

def constrained_region_sampler(s):
    # Unif(-5, 5), last dim should be 1
    return 10 * torch.rand(s, 1) - 5


'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''
all_experiments = []

prototype = {
    'title': '1D_test',
    'nn' : {
        'architecture' : [1, 20, 20, 1],
        'nonlinearity' : rbf,
        'prior_ds' : ds.Normal(0.0, 4.0),
    },
    'data' : {
        'ground_truth': ground_truth,
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-4, 4),
        'plt_y_domain': (-3, 12),
        'integral_constrained_region' : 10, 
        'X':  X,
        'Y':  Y,
        'X_plot': X_plot,
        'X_v_id': X_id,
        'X_v_ood': X_ood,
    },
    'constraints' : {
        'constr': constr,
        'plot' : [],
    },
    'bbb' :{
        'BbB_rv_samples': 20,
        'regular' : {
            'iterations' : 100,
            'restarts': 2,
            'reporting_every_' : 20,
            'cores_used' : 1,
        },
        'constrained': {
            'iterations': 100,
            'restarts': 1,
            'reporting_every_': 20,
            'cores_used': 1,
            'violation_samples' : 300,
            'tau_tuple': (30.0, 1.0),
            'gamma': 20000,
            'constrained_region_sampler': constrained_region_sampler,
        },
        'initialize_q' : {
            'mean': 1.0,  # * torch.randn
            'std' : -5.0, # * torch.ones
        },
        'posterior_predictive_analysis': {
            'posterior_samples': 30,
            'constrained_region_samples_for_pp_violation': 30,
        }

    },
    'experiment' : {
        'run_regular_BbB' : True,
        'run_constrained_BbB': True,
        'multithread_computation': False,
        'compute_held_out_loglik': True,
        'show_function_samples': True,
        'show_posterior_predictive': True,
        'show_plot_training_evaluations': True,
        'show_constraint_function_heatmap': True,
        'plot_size' : (6, 4), # width, height (inches)
    },
}

all_experiments.append(prototype)

# repeat 
# all_experiments = 1 * all_experiments




# returns inputs X s.t. x in X is from constrained X region
# def constrained_X_sampler(s):
#     # via rejection sampling
#     samples = []
#     while len(samples) < s:
#         z = 3 * np.random.normal(size=n_dim)
#         valid = True
#         for region in constr:
#             x_consts, _ = region
#             for constraint in x_consts:
#                 # in this ex., constr. is independent y
#                 if constraint.f(z, None) > 0:
#                     valid = False
#         if valid:
#             samples.append(z)

#     return np.array(samples)


