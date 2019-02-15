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
from main import main
from hmc import main_HMC


'''
Preliminary definitions
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


def rbf(x): return torch.exp(- x.pow(2))
relu = ReLUActivation.apply
def tanh(x): return x.tanh(x)
def softrelu(x): return torch.log(1.0 + x.exp())


# Data

N, n_dim = 10, 1

def ground_truth(x):
    return - x.pow(4) + 3 * x.pow(2) + 1
    
    
X = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).unsqueeze(1)
Y = ground_truth(X)

X_plot = torch.linspace(-5, 5, steps=100).unsqueeze(1)
Y_plot = ground_truth(X_plot)

X_id = torch.tensor([-1.9, -1.5, 0.5, 0.0, 0.5, 1.5, 1.9]).unsqueeze(1)
Y_id = ground_truth(X_id)

X_ood = torch.tensor([-4, -3, -2.5, 2.5, 3, 4]).unsqueeze(1)
Y_ood = ground_truth(X_ood)


'''
Constraints: f(x, y) <= 0

Have to be broadcastable!!
i.e. x is (datapoints, in_dim)
     y is (weights samples, datapoints, out_dim)


'''

# DON'T FORGET: 
#   1)  Double check constraint def.
#   2)  constaint sampler
#   3)  plot rectangle

# right
def x_c00(x, y): return x - 4
def x_c01(x, y): return - x + 3
def y_c02(x, y): return y - 1.5
def y_c03(x, y): return - y - 0.5

# center
def x_c10(x, y): return x - 0.5
def x_c11(x, y): return - x - 0.5
def y_c12(x, y): return y - 4.5
def y_c13(x, y): return - y + 2.5

# left
def x_c20(x, y): return x + 3
def x_c21(x, y): return - x - 4
def y_c22(x, y): return y + 1.5
def y_c23(x, y): return - y - 3.5
    

constr = [
    # ([x_c00, x_c01], [y_c02, y_c03]),
    ([x_c10, x_c11], [y_c12, y_c13]),
    # ([x_c20, x_c21], [y_c22, y_c23]),
]


def constrained_region_sampler(s):
    # return torch.cat([
    #     ds.Uniform(3, 4).sample(sample_shape=torch.Size([int(round(s / 3)), 1])),
    #     ds.Uniform(-4, -3).sample(sample_shape=torch.Size([int(round(s / 3)), 1])),
    #     ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([int(round(s / 3)), 1]))], dim=0)
    
    # 

    out = torch.tensor(
        [[ 0.1980],
        [-0.1096],
        [-0.1930],
        [-0.0334],
        [ 0.3723],
        [ 0.3721],
        [ 0.2059],
        [ 0.2735],
        [-0.3145],
        [ 0.3457],
        [-0.1102],
        [-0.3033],
        [-0.3982],
        [-0.2497],
        [ 0.0233],
        [-0.4887],
        [-0.1787],
        [-0.4214],
        [ 0.2368],
        [-0.4516],
        [-0.0983],
        [ 0.0203],
        [ 0.4832],
        [ 0.0577],
        [-0.3671],
        [-0.4158],
        [ 0.1174],
        [ 0.2331],
        [ 0.2632],
        [ 0.2690],
        [-0.3421],
        [ 0.2196],
        [ 0.0987],
        [-0.1941],
        [ 0.2327],
        [ 0.3242],
        [-0.1098],
        [ 0.4727],
        [ 0.3315],
        [ 0.0091],
        [-0.2158],
        [ 0.4184],
        [-0.0429],
        [-0.4295],
        [-0.1164],
        [-0.0657],
        [-0.0858],
        [ 0.4988],
        [-0.2519],
        [ 0.1875],
        [ 0.0294],
        [ 0.2891],
        [-0.0541],
        [-0.0266],
        [ 0.4544],
        [-0.1821],
        [ 0.0897],
        [-0.1230],
        [ 0.1435],
        [-0.2881],
        [ 0.0243],
        [-0.3369],
        [-0.4826],
        [-0.4797],
        [ 0.1385],
        [ 0.0867],
        [-0.0362],
        [-0.1271],
        [-0.3812],
        [ 0.3800],
        [-0.3352],
        [-0.1305],
        [-0.0009],
        [ 0.3329],
        [ 0.2565],
        [-0.2083],
        [ 0.0409],
        [ 0.2124],
        [-0.0566],
        [-0.0041],
        [-0.0968],
        [-0.1044],
        [-0.0967],
        [ 0.0277],
        [-0.3166],
        [-0.1538],
        [ 0.0975],
        [ 0.0390],
        [-0.4847],
        [ 0.4884],
        [ 0.2875],
        [-0.1317],
        [ 0.0531],
        [-0.3293],
        [-0.2538],
        [-0.1244],
        [ 0.3638],
        [-0.2406],
        [ 0.0360],
        [-0.4616]])
    out = ds.Uniform(-0.5, 0.5).sample(sample_shape=torch.Size([s, 1]))
    return out


'''Darting HMC Modes Preprocessed'''
darting_points = torch.tensor(
    [[7.7704e-01,  1.2386e+00, -7.8888e-13,  2.5510e+00,  5.9671e-01,
      5.9652e-01, -2.7594e-01,  4.2074e+00, -5.9692e-01, -1.3641e+00,
      5.9678e-01, -2.3970e+00, -1.9175e+00, -1.0935e+01,  5.9707e-01,
      -2.3146e+00, -2.9656e+00,  7.1216e-12,  5.5665e+00,  1.4502e-01,
      1.4511e-01,  8.0792e-01, -1.7650e+00, -1.4447e-01, -1.9905e-01,
      1.4491e-01,  3.8161e+00, -1.7467e+00,  9.1801e+00,  1.4512e-01,
      -8.8306e-01, -6.3485e+00, -1.5509e+00, -7.0495e+00,  7.4778e-01,
      7.4334e-01, -1.4037e+00,  7.5040e+00,  7.5299e-01,  8.9488e-01,
      7.4952e-01,  3.8273e+00, -1.5791e+00, -4.8446e-02,  7.5398e-01,
      3.2316e+00],
        [1.5484e-01,  5.5499e-01,  1.6852e+00, -2.4235e+00, -6.7358e+00,
         6.0172e-01,  6.8605e-01, -4.2734e+00, -5.9630e-01, -1.6857e+00,
         5.8160e-01, -1.3940e+00,  2.0208e-15, -9.8160e+00,  1.5770e+00,
         7.2734e-01, -1.4308e+00,  3.8547e+00, -7.2019e+00,  8.5721e+00,
         -1.0889e-02, -1.6979e+00,  7.1916e+00,  4.4814e-03, -2.4488e+00,
         1.1214e-02, -3.3899e-01, -3.4253e-14, -7.5692e+00, -1.6852e+00,
         -3.6925e-01, -1.2541e+00, -4.7855e+00,  2.4984e+01, -1.0104e+01,
         2.4143e+00, -1.5212e+00,  3.3208e+00,  1.7318e+00,  5.5094e+00,
         9.3536e-01, -2.0712e+00, -8.2360e+00, -1.1407e+01,  3.7585e+00,
         5.7125e+00],
        [7.7704e-01,  1.2386e+00, -7.8888e-13,  2.5510e+00,  5.9671e-01,
         5.9652e-01, -2.7594e-01,  4.2074e+00, -5.9692e-01, -1.3641e+00,
         5.9678e-01, -2.3970e+00, -1.9175e+00, -1.0935e+01,  5.9707e-01,
         -2.3146e+00, -2.9656e+00,  7.1216e-12,  5.5665e+00,  1.4502e-01,
         1.4511e-01,  8.0792e-01, -1.7650e+00, -1.4447e-01, -1.9905e-01,
         1.4491e-01,  3.8161e+00, -1.7467e+00,  9.1801e+00,  1.4512e-01,
         -8.8306e-01, -6.3485e+00, -1.5509e+00, -7.0495e+00,  7.4778e-01,
         7.4334e-01, -1.4037e+00,  7.5040e+00,  7.5299e-01,  8.9488e-01,
         7.4952e-01,  3.8273e+00, -1.5791e+00, -4.8446e-02,  7.5398e-01,
         3.2316e+00],
        [4.5278e+00, -1.9442e+00, -1.4516e+00, -4.6932e+00,  1.4073e-02,
         5.5046e-01, -1.6168e+00,  3.7162e-02, -5.5077e-01, -8.9454e-01,
         -5.5061e-01, -4.4871e-03, -5.5035e-01, -6.4762e-19,  2.4908e+00,
         9.2806e+00,  2.8346e+00, -1.7568e+00, -7.2395e+00, -1.2396e-01,
         4.1094e-02,  2.1986e+00,  1.2177e+00, -3.7441e-02, -8.8147e-03,
         -4.3413e-02,  3.9463e-02, -4.4066e-02, -4.9115e+00, -5.3632e+00,
         -3.3877e+00,  2.6302e+00,  2.1260e+00,  9.7023e-01, -6.5817e-01,
         8.5062e-01,  1.4189e+00,  5.6486e-01,  8.5363e-01, -2.1693e-01,
         8.5850e-01, -6.5731e-01,  8.4980e-01, -9.0472e-11, -4.3599e+00,
         -3.2334e-01],
        [-1.0078e+00, -1.5575e+00,  1.0078e+00, -1.5640e+00,  1.0082e+00,
         -8.3062e+00, -1.0079e+00, -6.7993e+00,  1.0077e+00,  1.6829e+00,
         -7.5190e+00, -3.9621e+00,  4.0944e+00,  1.0078e+00,  1.6458e+00,
         8.1714e-02,  1.2822e+00, -8.1733e-02,  1.2950e+00, -8.2871e-02,
         5.8330e+00,  8.1921e-02, -7.3138e+00, -8.1697e-02,  7.7648e-01,
         -7.8209e+00, -6.4930e+00, -6.6737e+00, -8.1714e-02,  1.0724e+00,
         1.1429e+00,  2.0787e+00,  1.1429e+00,  2.0995e+00,  1.1430e+00,
         2.9887e+00,  1.1419e+00, -5.8253e+00,  1.1427e+00,  6.1777e-01,
         7.2468e+00,  5.6115e+00,  5.2943e+00,  1.1429e+00,  3.1594e+00,
         -3.7903e+00],
        [1.6707e-06,  2.3229e-20, -3.7205e+00, -7.3765e-11,  8.8828e-01,
         -1.0215e+00, -5.9526e+00,  1.4987e+00,  8.0535e-17, -1.2309e+00,
         1.3364e+00,  3.7799e-02,  4.1131e-01,  2.7939e+00, -1.4833e-20,
         -1.4628e-08, -4.8095e-20, -7.7430e+00,  1.1809e-11, -1.0335e+00,
         2.7134e+00,  1.2016e+01,  2.0339e+00, -2.4603e-17, -2.8533e+00,
         -2.0349e-01,  7.4315e-02, -2.6651e-01,  5.9502e+00,  2.9546e-20,
         1.9247e-02,  4.4865e-02, -2.7524e+00,  4.0451e-02,  1.3165e+00,
         -5.2253e+00, -2.5795e+00,  1.4590e+00,  4.7525e-02, -1.0491e+00,
         -1.0838e+00, -8.7539e-02,  7.4281e-01, -2.2550e+00,  4.4550e-02,
         1.4591e+00],
        [7.7704e-01,  1.2386e+00, -7.8888e-13,  2.5510e+00,  5.9671e-01,
         5.9652e-01, -2.7594e-01,  4.2074e+00, -5.9692e-01, -1.3641e+00,
         5.9678e-01, -2.3970e+00, -1.9175e+00, -1.0935e+01,  5.9707e-01,
         -2.3146e+00, -2.9656e+00,  7.1216e-12,  5.5665e+00,  1.4502e-01,
         1.4511e-01,  8.0792e-01, -1.7650e+00, -1.4447e-01, -1.9905e-01,
         1.4491e-01,  3.8161e+00, -1.7467e+00,  9.1801e+00,  1.4512e-01,
         -8.8306e-01, -6.3485e+00, -1.5509e+00, -7.0495e+00,  7.4778e-01,
         7.4334e-01, -1.4037e+00,  7.5040e+00,  7.5299e-01,  8.9488e-01,
         7.4952e-01,  3.8273e+00, -1.5791e+00, -4.8446e-02,  7.5398e-01,
         3.2316e+00],
        [7.7704e-01,  1.2386e+00, -7.8888e-13,  2.5510e+00,  5.9671e-01,
         5.9652e-01, -2.7594e-01,  4.2074e+00, -5.9692e-01, -1.3641e+00,
         5.9678e-01, -2.3970e+00, -1.9175e+00, -1.0935e+01,  5.9707e-01,
         -2.3146e+00, -2.9656e+00,  7.1216e-12,  5.5665e+00,  1.4502e-01,
         1.4511e-01,  8.0792e-01, -1.7650e+00, -1.4447e-01, -1.9905e-01,
         1.4491e-01,  3.8161e+00, -1.7467e+00,  9.1801e+00,  1.4512e-01,
         -8.8306e-01, -6.3485e+00, -1.5509e+00, -7.0495e+00,  7.4778e-01,
         7.4334e-01, -1.4037e+00,  7.5040e+00,  7.5299e-01,  8.9488e-01,
         7.4952e-01,  3.8273e+00, -1.5791e+00, -4.8446e-02,  7.5398e-01,
         3.2316e+00],
        [7.7704e-01,  1.2386e+00, -7.8888e-13,  2.5510e+00,  5.9671e-01,
         5.9652e-01, -2.7594e-01,  4.2074e+00, -5.9692e-01, -1.3641e+00,
         5.9678e-01, -2.3970e+00, -1.9175e+00, -1.0935e+01,  5.9707e-01,
         -2.3146e+00, -2.9656e+00,  7.1216e-12,  5.5665e+00,  1.4502e-01,
         1.4511e-01,  8.0792e-01, -1.7650e+00, -1.4447e-01, -1.9905e-01,
         1.4491e-01,  3.8161e+00, -1.7467e+00,  9.1801e+00,  1.4512e-01,
         -8.8306e-01, -6.3485e+00, -1.5509e+00, -7.0495e+00,  7.4778e-01,
         7.4334e-01, -1.4037e+00,  7.5040e+00,  7.5299e-01,  8.9488e-01,
         7.4952e-01,  3.8273e+00, -1.5791e+00, -4.8446e-02,  7.5398e-01,
         3.2316e+00],
        [7.7704e-01,  1.2386e+00, -7.8888e-13,  2.5510e+00,  5.9671e-01,
         5.9652e-01, -2.7594e-01,  4.2074e+00, -5.9692e-01, -1.3641e+00,
         5.9678e-01, -2.3970e+00, -1.9175e+00, -1.0935e+01,  5.9707e-01,
         -2.3146e+00, -2.9656e+00,  7.1216e-12,  5.5665e+00,  1.4502e-01,
         1.4511e-01,  8.0792e-01, -1.7650e+00, -1.4447e-01, -1.9905e-01,
         1.4491e-01,  3.8161e+00, -1.7467e+00,  9.1801e+00,  1.4512e-01,
         -8.8306e-01, -6.3485e+00, -1.5509e+00, -7.0495e+00,  7.4778e-01,
         7.4334e-01, -1.4037e+00,  7.5040e+00,  7.5299e-01,  8.9488e-01,
         7.4952e-01,  3.8273e+00, -1.5791e+00, -4.8446e-02,  7.5398e-01,
         3.2316e+00]])

'''
Experiment dictionary 

Constraints are of the form
    [region_0, region_1, ...]
    with region = ([x_constraint_0, ..], [y_constraint_0, ..])

'''
all_experiments = []


prototype = {
    'title': '6pt_toy_example',
    'nn': {
        'architecture': [n_dim, 15, 1],
        'nonlinearity': rbf,
        'prior_ds': ds.Normal(0.0, 3.0),
    },
    'data': {
        'noise_ds': ds.Normal(0.0, 0.1),
        'plt_x_domain': (-5, 5),
        'plt_y_domain': (-12, 12),
        'integral_constrained_region': 0,
        'X':  X,
        'Y':  Y,
        'X_plot': X_plot,
        'Y_plot': Y_plot,
        'X_v_id': X_id,
        'Y_v_id': Y_id,
        'X_v_ood': X_ood,
        'Y_v_ood': Y_ood,
    },
    'constraints': {
        'constr': constr,
        'plot': [
            # DrawRectangle(bottom_left=(3, -0.5), top_right=(4, 1.5)),
            DrawRectangle(bottom_left=(-0.5, 2.5), top_right=(0.5, 4.5)),
            # DrawRectangle(bottom_left=(-4, -3.5), top_right=(-3, -1.5)),
        ],
    },
    'bbb': {
        'BbB_rv_samples': 100,
        'batch_size': 0,  # batch_size = 0 implies full dataset training
        'regular': {
            'iterations': 5000,
            'restarts': 4,
            'reporting_every_': 50,
            'cores_used': 1,
        },
        'constrained': {
            'iterations': 1000,
            'restarts': 1,
            'reporting_every_': 50,
            'cores_used': 1,
            'violation_samples': 500,
            'tau_tuple': (10.0, 1.0),
            'gamma': 50,
            'constrained_region_sampler': constrained_region_sampler,
        },
        'initialize_q': {
            'mean': 1.0,  # * torch.randn
            'std': -3.0,  # * torch.ones
        },
        'posterior_predictive_analysis': {
            'posterior_samples': 50,
            'constrained_region_samples_for_pp_violation': 50,
        }

    },
    'hmc' : {
        'darting' : {
            'bool' : True,
            'preprocessing' : {
                'bool' : False,
                'show_mode_cluster_analysis': False,
                'searched_modes': 40,
                'mode_searching_convergence': 0.005,
                'n_darting_regions': 10,
                'preprocessed_regions': darting_points,
            },
            'algorithm' : {
                'darting_region_radius' : 2.0e1,  
                'p_check' : 0.03,
            },
        },
        'stepsize' : 0.01,
        'steps' : 30,
        'hmc_samples' : 4000,
        'burnin' : 0,
        'thinning' : 3,
    },
    'experiment': {
        'run_regular_BbB': True,
        'run_constrained_BbB': False,
        'multithread_computation': False,
        'compute_held_out_loglik_id': True,
        'compute_held_out_loglik_ood': False,
        'compute_RMSE_id': True,
        'compute_RMSE_ood': False,
        'show_function_samples': True,
        'show_posterior_predictive': True,
        'show_plot_training_evaluations': True,
        'show_constraint_function_heatmap': False,
        'plot_size': (6, 4),  # width, height (inches)
    },
}


all_experiments.append(prototype)

# main(all_experiments)
main_HMC(all_experiments)




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


